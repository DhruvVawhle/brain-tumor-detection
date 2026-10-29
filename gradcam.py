"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation for PyTorch models.
Provides interpretability maps and visual overlays for MRI brain tumor classification.
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM:
    """
    Computes Grad-CAM visualizations by intercepting activations and backward gradients
    at a specific convolutional target layer.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            if grad_output and len(grad_output) > 0 and grad_output[0] is not None:
                self.gradients = grad_output[0].detach()

        # Register forward and backward hooks
        h_fwd = self.target_layer.register_forward_hook(forward_hook)
        if hasattr(self.target_layer, "register_full_backward_hook"):
            h_bwd = self.target_layer.register_full_backward_hook(backward_hook)
        else:
            h_bwd = self.target_layer.register_backward_hook(backward_hook)

        self.handles.extend([h_fwd, h_bwd])

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int = None) -> tuple[np.ndarray, int]:
        """
        Generate normalized 2D Grad-CAM heatmap array in range [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()

        # Ensure tensor is shaped [1, 3, H, W]
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = int(torch.argmax(output, dim=1).item())

        score = output[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Grad-CAM could not capture activations or gradients. "
                "Ensure target layer is placed before pooling/flattening."
            )

        # Global average pooling across spatial dimensions of gradients
        # Shape: [1, C, H, W] -> [C]
        weights = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weighted combination of forward activation maps
        # Copy to avoid modifying cached activations
        act = self.activations.clone()
        for i in range(act.shape[1]):
            act[:, i, :, :] *= weights[i]

        # Apply ReLU to retain only features that positively influence the score
        cam = torch.sum(act, dim=1).squeeze()
        cam = torch.clamp(cam, min=0.0)

        cam_np = cam.cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = np.min(cam_np), np.max(cam_np)
        if cam_max > cam_min:
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_np)

        return cam_norm, target_class

    def remove_hooks(self):
        """Clean up PyTorch hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def get_target_layer_for_model(model: nn.Module, model_name: str) -> nn.Module:
    """
    Selects the most suitable convolutional layer for Grad-CAM depending on architecture.
    """
    name = model_name.strip().lower()

    if "resnet" in name:
        # Final residual block in layer 4
        if hasattr(model, "layer4"):
            return model.layer4[-1]
        raise AttributeError("ResNet model is missing layer4 attribute.")

    elif "mobilenet" in name:
        # Final feature extraction block before pooling
        if hasattr(model, "features"):
            return model.features[-1]
        raise AttributeError("MobileNetV3 is missing features attribute.")

    elif "custom" in name or "cnn" in name:
        # Traverse backward to find the last Conv2d or last module in features
        if hasattr(model, "features"):
            for layer in reversed(model.features):
                if isinstance(layer, nn.Conv2d):
                    return layer
            return model.features[-1]
        raise AttributeError("Custom CNN is missing features attribute.")

    else:
        # Generic fallback: search for last Conv2d layer
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is not None:
            return last_conv
        raise ValueError(f"Could not locate an appropriate target layer for {model_name}.")


def apply_gradcam_overlay(
    original_pil: Image.Image,
    cam_mask: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "jet"
) -> tuple[Image.Image, Image.Image]:
    """
    Applies the 2D Grad-CAM mask over the original image.
    Returns:
        (heatmap_pil, blended_pil)
    """
    # Resize original image to 224x224 standard display
    img_rgb = original_pil.convert("RGB").resize((224, 224))
    img_np = np.array(img_rgb, dtype=np.float32) / 255.0

    # Resize CAM mask to 224x224 using PIL for smooth bicubic interpolation
    cam_pil = Image.fromarray(cam_mask.astype(np.float32))
    cam_resized = np.array(cam_pil.resize((224, 224), Image.Resampling.BICUBIC))
    cam_resized = np.clip(cam_resized, 0.0, 1.0)

    # Colorize using matplotlib colormap
    try:
        cmap = plt.get_cmap(colormap_name)
    except Exception:
        cmap = plt.get_cmap("jet")

    heatmap_colored = cmap(cam_resized)[:, :, :3]  # Strip alpha if present (RGB)

    # Overlay: alpha * heatmap + (1 - alpha) * original
    blended = (1.0 - alpha) * img_np + alpha * heatmap_colored
    blended = np.clip(blended, 0.0, 1.0)

    heatmap_img = Image.fromarray((heatmap_colored * 255).astype(np.uint8))
    blended_img = Image.fromarray((blended * 255).astype(np.uint8))

    return heatmap_img, blended_img

