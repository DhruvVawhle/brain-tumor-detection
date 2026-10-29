"""
Brain Tumor Detection from MRI Images
Data Science Honours Project - Streamlit Web Application
Compatible with Google Colab and Local Deployments
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


# =========================================================
# 1. MODEL ARCHITECTURES & DEFINITIONS
# =========================================================

class CustomCNN(nn.Module):
    """
    Standard 4-block Convolutional Neural Network designed for 224x224 MRI images.
    Features AdaptiveAvgPool2d to ensure compatibility with varied spatial dimensions.
    """
    def __init__(self, num_classes=4):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112x112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56x56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 14x14
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(model_name: str, num_classes: int = 4) -> nn.Module:
    """Builds base model architecture configured for num_classes."""
    name = model_name.strip().lower()

    if "resnet18" in name:
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    elif "mobilenetv3" in name or "mobilenet" in name:
        # Default to MobileNetV3-Large, with fallback capacity
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model

    elif "custom" in name or "cnn" in name:
        return CustomCNN(num_classes=num_classes)

    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")


def clean_state_dict(checkpoint):
    """Extract raw state_dict and remove DataParallel / container prefixes if present."""
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned = {}
    for k, v in state_dict.items():
        # Remove DataParallel or DistributedDataParallel prefixes
        new_key = k.replace("module.", "").replace("model.", "")
        cleaned[new_key] = v
    return cleaned


def load_model_weights(model: nn.Module, weights_path: str, device: str = "cpu") -> tuple[nn.Module, str]:
    """
    Safely load weights into model. Handles device mapping and provides detailed status.
    Returns (model, message).
    """
    if not os.path.exists(weights_path):
        return model, f"Weight file not found: {weights_path}"

    try:
        checkpoint = torch.load(weights_path, map_location=torch.device(device))

        # Check if saved as entire serialized nn.Module
        if isinstance(checkpoint, nn.Module):
            model = checkpoint
            model.to(device)
            model.eval()
            return model, "Loaded successfully as full serialized PyTorch model."

        cleaned_dict = clean_state_dict(checkpoint)

        # Attempt strict loading
        try:
            model.load_state_dict(cleaned_dict, strict=True)
            model.to(device)
            model.eval()
            return model, "Weights loaded successfully (strict match)."
        except RuntimeError:
            # Fallback to non-strict matching if slight layer mismatches exist
            missing, unexpected = model.load_state_dict(cleaned_dict, strict=False)
            model.to(device)
            model.eval()
            return model, f"Weights loaded (non-strict). Missing: {len(missing)}, Unexpected: {len(unexpected)}."

    except Exception as e:
        return model, f"Error loading weights: {str(e)}"


def get_preprocessing_transforms(image_size=(224, 224)):
    """
    Standard preprocessing transforms matching PyTorch training pipeline on MRI scans.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def safe_open_image(uploaded_file) -> Image.Image:
    """
    Safely opens any image format (JPG, PNG, WEBP, BMP, TIFF, GIF, PPM, ICO, etc.),
    normalizes EXIF rotation, composites transparent alpha channels onto a neutral white background,
    and returns a standardized 3-channel RGB PIL Image.
    """
    try:
        img = Image.open(uploaded_file)

        # Correct EXIF rotation if present (e.g. smartphone/scanned photos)
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        # Handle transparency or palette modes gracefully
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            rgba = img.convert("RGBA")
            bg.paste(rgba, mask=rgba.split()[3])
            return bg
        elif img.mode != "RGB":
            return img.convert("RGB")
        return img
    except Exception as e:
        raise ValueError(f"Unable to decode file as an image. Please ensure it is a valid image format. Details: {e}")


# =========================================================
# 2. GRAD-CAM IMPLEMENTATION
# =========================================================

class GradCAM:
    """
    PyTorch hook-based Grad-CAM computation.
    Extracts spatial feature maps and class-specific gradients to produce saliency maps.
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

        h_fwd = self.target_layer.register_forward_hook(forward_hook)
        if hasattr(self.target_layer, "register_full_backward_hook"):
            h_bwd = self.target_layer.register_full_backward_hook(backward_hook)
        else:
            h_bwd = self.target_layer.register_backward_hook(backward_hook)

        self.handles.extend([h_fwd, h_bwd])

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int = None) -> tuple[np.ndarray, int]:
        self.model.eval()
        self.model.zero_grad()

        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = int(torch.argmax(output, dim=1).item())

        score = output[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations were not intercepted by the hooks.")

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weighted combination of activation maps
        act = self.activations.clone()
        for i in range(act.shape[1]):
            act[:, i, :, :] *= weights[i]

        cam = torch.sum(act, dim=1).squeeze()
        cam = torch.clamp(cam, min=0.0)

        cam_np = cam.cpu().numpy()

        cam_min, cam_max = np.min(cam_np), np.max(cam_np)
        if cam_max > cam_min:
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_np)

        return cam_norm, target_class

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def get_target_layer_for_model(model: nn.Module, model_name: str) -> nn.Module:
    """Selects the final convolutional feature extraction layer for Grad-CAM."""
    name = model_name.strip().lower()

    if "resnet" in name:
        if hasattr(model, "layer4"):
            return model.layer4[-1]
        raise AttributeError("ResNet is missing layer4.")

    elif "mobilenet" in name:
        if hasattr(model, "features"):
            return model.features[-1]
        raise AttributeError("MobileNet is missing features.")

    elif "custom" in name or "cnn" in name:
        if hasattr(model, "features"):
            for layer in reversed(model.features):
                if isinstance(layer, nn.Conv2d):
                    return layer
            return model.features[-1]
        raise AttributeError("Custom CNN is missing features.")

    else:
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is not None:
            return last_conv
        raise ValueError(f"Could not locate target layer for {model_name}.")


def apply_gradcam_overlay(
    original_pil: Image.Image,
    cam_mask: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "jet"
) -> tuple[Image.Image, Image.Image]:
    """Applies 2D Grad-CAM mask over original image with smooth color map."""
    img_rgb = original_pil.convert("RGB").resize((224, 224))
    img_np = np.array(img_rgb, dtype=np.float32) / 255.0

    cam_pil = Image.fromarray(cam_mask.astype(np.float32))
    cam_resized = np.array(cam_pil.resize((224, 224), Image.Resampling.BICUBIC))
    cam_resized = np.clip(cam_resized, 0.0, 1.0)

    try:
        cmap = plt.get_cmap(colormap_name)
    except Exception:
        cmap = plt.get_cmap("jet")

    heatmap_colored = cmap(cam_resized)[:, :, :3]
    blended = (1.0 - alpha) * img_np + alpha * heatmap_colored
    blended = np.clip(blended, 0.0, 1.0)

    heatmap_img = Image.fromarray((heatmap_colored * 255).astype(np.uint8))
    blended_img = Image.fromarray((blended * 255).astype(np.uint8))

    return heatmap_img, blended_img


# =========================================================
# 3. STREAMLIT APPLICATION CONFIG & CACHES
# =========================================================

st.set_page_config(
    page_title="Brain Tumor Detection | Data Science Honours",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 12px;
    }
    .disclaimer-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 0.9rem;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_class_names(json_path: str = "class_names.json") -> list[str]:
    """Reads class names dynamically from JSON or applies fallback."""
    candidates = [
        json_path,
        os.path.join("results", json_path),
        os.path.join("models", json_path)
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        return data
                    elif isinstance(data, dict):
                        return [data[str(i)] for i in range(len(data))]
            except Exception:
                pass
    return ["glioma", "meningioma", "notumor", "pituitary"]


@st.cache_resource
def load_model_cached(model_name: str, num_classes: int):
    """Caches loaded model weights in memory with multi-path resolution and variant auto-detection."""
    filename_map = {
        "Custom CNN": "custom_cnn.pth",
        "ResNet18": "resnet18.pth",
        "MobileNetV3": "mobilenetv3.pth"
    }
    filename = filename_map.get(model_name, "custom_cnn.pth")
    candidates = [
        filename,
        os.path.join("models", filename),
        os.path.join("results", filename)
    ]

    weights_path = None
    for path in candidates:
        if os.path.exists(path):
            weights_path = path
            break

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Auto-detect MobileNetV3 small vs large variant from checkpoint
    if "mobilenet" in model_name.lower() and weights_path:
        try:
            ckpt = torch.load(weights_path, map_location="cpu")
            cdict = clean_state_dict(ckpt)
            if any("features.15" in k for k in cdict.keys()):
                base_model = models.mobilenet_v3_large(weights=None)
            else:
                base_model = models.mobilenet_v3_small(weights=None)
            in_features = base_model.classifier[3].in_features
            base_model.classifier[3] = nn.Linear(in_features, num_classes)
        except Exception:
            base_model = build_model(model_name, num_classes=num_classes)
    else:
        base_model = build_model(model_name, num_classes=num_classes)

    if weights_path is not None:
        model, msg = load_model_weights(base_model, weights_path, device=device)
        is_loaded = "successfully" in msg.lower() or "strict" in msg.lower()
        return model, is_loaded, msg, device
    else:
        return base_model, False, f"File `{filename}` not found in working directory, `models/`, or `results/`.", device


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("🧠 NeuroScan AI")
st.sidebar.caption("Data Science Honours Project")

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "🩺 MRI Inference",
        "🔍 Grad-CAM Explainability",
        "📊 Model Comparison",
        "ℹ️ Project Information"
    ]
)

classes = load_class_names()
num_classes = len(classes)

st.sidebar.markdown("---")
st.sidebar.subheader("File Verification")
file_status = {
    "custom_cnn.pth": "Custom CNN",
    "resnet18.pth": "ResNet18",
    "mobilenetv3.pth": "MobileNetV3",
    "class_names.json": "Class Names",
    "model_comparison.csv": "Metrics CSV"
}
for f, label in file_status.items():
    exists = os.path.exists(f) or os.path.exists(os.path.join("models", f)) or os.path.exists(os.path.join("results", f))
    if exists:
        st.sidebar.markdown(f"✅ **{label}**: Found")
    else:
        st.sidebar.markdown(f"⚠️ **{label}**: Missing")

st.sidebar.markdown("---")
st.sidebar.caption(f"Runtime Device: **{'CUDA GPU' if torch.cuda.is_available() else 'CPU'}**")


def render_medical_disclaimer():
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Academic & Scientific Demonstration Notice:</strong><br>
        This application was engineered solely for an academic Data Science Honours project.
        It is <strong>NOT</strong> an accredited diagnostic tool. Deep learning predictions and Grad-CAM maps
        must never substitute for clinical judgment or certified radiologist review.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# PAGE 1: MRI INFERENCE
# =========================================================
if nav_choice == "🩺 MRI Inference":
    st.header("Brain Tumor MRI Classification")
    st.write("Upload a brain MRI scan (Axial, Coronal, or Sagittal view) to run deep learning inference.")
    render_medical_disclaimer()

    col_ctrl, col_view = st.columns([1, 1], gap="medium")

    with col_ctrl:
        st.subheader("1. Model & Input Selection")
        selected_model_name = st.selectbox(
            "Select Deep Learning Model",
            ["ResNet18", "MobileNetV3", "Custom CNN"],
            index=0
        )

        model, is_loaded, status_msg, device = load_model_cached(selected_model_name, num_classes)
        if is_loaded:
            st.success(f"Loaded `{selected_model_name}` ({status_msg})")
        else:
            st.warning(f"Using uninitialized weights for `{selected_model_name}`: {status_msg}")

        uploaded_file = st.file_uploader(
            "Upload Brain MRI Scan (All image formats accepted)",
            type=None,
            help="Accepts any image format: JPG, JPEG, PNG, WEBP, BMP, TIFF, GIF, PPM, PGM, PNM, ICO, etc."
        )

    with col_view:
        st.subheader("2. MRI Preview")
        if uploaded_file is not None:
            try:
                pil_image = safe_open_image(uploaded_file)
                st.image(pil_image, caption=f"Scan: {uploaded_file.name}", use_container_width=True)
                st.caption(f"Dimensions: {pil_image.size[0]} x {pil_image.size[1]} px | Format: {getattr(uploaded_file, 'type', 'Image') or 'Image'}")
            except Exception as e:
                st.error(f"Error opening image: {e}")
                pil_image = None
        else:
            st.info("Please upload an MRI image in any image format (JPG, PNG, WEBP, TIFF, BMP, etc.).")
            pil_image = None

    if pil_image is not None:
        st.markdown("---")
        st.subheader("3. Model Diagnostic Predictions")

        transform = get_preprocessing_transforms(image_size=(224, 224))
        input_tensor = transform(pil_image).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = F.softmax(logits, dim=1).squeeze().cpu().numpy()

        predicted_idx = int(np.argmax(probabilities))
        predicted_class = classes[predicted_idx]
        confidence = float(probabilities[predicted_idx]) * 100.0

        col_res1, col_res2 = st.columns([1, 1], gap="medium")

        with col_res1:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin: 0; color: #555;">Predicted Classification</h4>
                <h2 style="margin: 6px 0; color: #1f77b4;">{predicted_class.upper()}</h2>
                <p style="font-size: 1.1rem; margin: 0;"><strong>Confidence:</strong> {confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

            class_descriptions = {
                "glioma": "Gliomas arise from glial cells. They often present with invasive, irregular infiltration on T1/T2 MRI.",
                "meningioma": "Meningiomas arise from arachnoid cap cells of the meninges; typically extra-axial and well-demarcated.",
                "pituitary": "Pituitary adenomas originate in the sella turcica, impacting endocrine functions and visual tracts.",
                "notumor": "Control MRI scan demonstrating preserved anatomical symmetry without focal neoplastic mass."
            }
            desc = class_descriptions.get(predicted_class.lower(), "Classified according to dataset labels.")
            st.info(f"**Pathological Overview:** {desc}")

        with col_res2:
            st.write("**Probability Distribution across Classes:**")
            prob_df = pd.DataFrame({
                "Class": [c.capitalize() for c in classes],
                "Probability (%)": [p * 100.0 for p in probabilities]
            }).sort_values(by="Probability (%)", ascending=False)

            fig, ax = plt.subplots(figsize=(6, 3.2))
            sorted_classes = prob_df["Class"].tolist()
            sorted_probs = prob_df["Probability (%)"].tolist()

            y_pos = np.arange(len(sorted_classes))
            ax.barh(y_pos, sorted_probs, color="#2b5c8f", align="center")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_classes)
            ax.invert_yaxis()
            ax.set_xlabel("Probability (%)")
            ax.set_xlim(0, 100)
            for i, v in enumerate(sorted_probs):
                ax.text(v + 1.5, i + 0.1, f"{v:.1f}%", color="#333", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


# =========================================================
# PAGE 2: GRAD-CAM EXPLAINABILITY
# =========================================================
elif nav_choice == "🔍 Grad-CAM Explainability":
    st.header("Grad-CAM Model Explainability")
    st.write(
        "Gradient-weighted Class Activation Mapping (Grad-CAM) highlights spatial features "
        "within the MRI scan that most significantly influenced the model's prediction."
    )
    render_medical_disclaimer()

    col_g_ctrl, col_g_img = st.columns([1, 1], gap="medium")

    with col_g_ctrl:
        st.subheader("Configuration")
        cam_model_name = st.selectbox(
            "Select Model Architecture",
            ["ResNet18", "MobileNetV3", "Custom CNN"],
            index=0,
            key="cam_model_select"
        )

        model, is_loaded, status_msg, device = load_model_cached(cam_model_name, num_classes)
        if not is_loaded:
            st.warning(f"Model `{cam_model_name}` using uninitialized weights: {status_msg}")

        uploaded_cam_file = st.file_uploader(
            "Upload MRI Image for Interpretability (All image formats accepted)",
            type=None,
            key="cam_uploader",
            help="Accepts any image format: JPG, JPEG, PNG, WEBP, BMP, TIFF, GIF, PPM, PGM, PNM, ICO, etc."
        )

        alpha_val = st.slider(
            "Heatmap Overlay Opacity (Alpha)",
            min_value=0.1,
            max_value=0.9,
            value=0.45,
            step=0.05
        )

        cmap_choice = st.selectbox(
            "Colormap",
            ["jet", "viridis", "inferno", "plasma"],
            index=0
        )

    with col_g_img:
        if uploaded_cam_file is not None:
            try:
                pil_cam_img = safe_open_image(uploaded_cam_file)
                st.image(pil_cam_img, caption=f"Source Scan: {uploaded_cam_file.name}", use_container_width=True)
            except Exception as e:
                st.error(f"Error opening image: {e}")
                pil_cam_img = None
        else:
            st.info("Upload an MRI scan in any image format to generate the Grad-CAM heatmap.")
            pil_cam_img = None

    if pil_cam_img is not None:
        st.markdown("---")
        st.subheader("Grad-CAM Visualizations")

        try:
            target_layer = get_target_layer_for_model(model, cam_model_name)
            gradcam_engine = GradCAM(model, target_layer)

            transform = get_preprocessing_transforms((224, 224))
            tensor_in = transform(pil_cam_img).unsqueeze(0).to(device)

            cam_mask, target_class_idx = gradcam_engine.generate_cam(tensor_in)
            gradcam_engine.remove_hooks()

            heatmap_pil, overlay_pil = apply_gradcam_overlay(
                pil_cam_img,
                cam_mask,
                alpha=alpha_val,
                colormap_name=cmap_choice
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.write("**1. Original Scan (224x224)**")
                st.image(pil_cam_img.resize((224, 224)), use_container_width=True)

            with c2:
                st.write(f"**2. Activation Heatmap ({cmap_choice})**")
                st.image(heatmap_pil, use_container_width=True)

            with c3:
                st.write(f"**3. Grad-CAM Overlay (α={alpha_val})**")
                st.image(overlay_pil, use_container_width=True)

            predicted_label = classes[target_class_idx].capitalize()
            st.success(f"Computed activation gradients for target class: **{predicted_label}**")

            st.markdown("""
            > **Interpretation Note:** 
            > Warm/red regions correspond to high activation gradients toward the predicted class.
            > Grad-CAM offers deep learning decision attribution, but is not an exact anatomical segmentation mask.
            """)

        except Exception as err:
            st.error(f"Grad-CAM computation encountered an issue: {err}")


# =========================================================
# PAGE 3: MODEL COMPARISON DASHBOARD
# =========================================================
elif nav_choice == "📊 Model Comparison":
    st.header("Comparative Model Performance Dashboard")
    st.write(
        "Evaluation metrics across models trained on the `masoudnickparvar/brain-tumor-mri-dataset` test set."
    )
    render_medical_disclaimer()

    csv_paths = [
        "model_comparison.csv",
        os.path.join("results", "model_comparison.csv"),
        os.path.join("models", "model_comparison.csv")
    ]
    df_metrics = None

    for p in csv_paths:
        if os.path.exists(p):
            try:
                df_metrics = pd.read_csv(p)
                break
            except Exception as e:
                st.error(f"Error loading {p}: {e}")

    if df_metrics is not None and not df_metrics.empty:
        st.subheader("Evaluation Metrics Table")
        st.dataframe(df_metrics, use_container_width=True)

        acc_col = None
        for col in ["Accuracy", "accuracy", "Acc"]:
            if col in df_metrics.columns:
                acc_col = col
                break

        if acc_col:
            best_idx = df_metrics[acc_col].astype(float).idxmax()
            best_row = df_metrics.loc[best_idx]
            best_model_name = best_row.get("Model", "Top Model")
            best_acc = float(best_row[acc_col]) * (100.0 if float(best_row[acc_col]) <= 1.0 else 1.0)
            st.success(f"🏆 **Top Performing Model:** **{best_model_name}** ({best_acc:.2f}% Accuracy)")

        st.subheader("Metric Comparison Chart")
        metric_cols = [c for c in df_metrics.columns if c.lower() not in ["model", "test_loss", "loss"]]

        if len(metric_cols) > 0 and "Model" in df_metrics.columns:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            x = np.arange(len(df_metrics["Model"]))
            total_bars = len(metric_cols)
            width = 0.8 / total_bars
            palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

            for i, col in enumerate(metric_cols):
                vals = df_metrics[col].astype(float).values
                if np.max(vals) <= 1.0:
                    vals = vals * 100.0
                offset = (i - total_bars / 2) * width + width / 2
                ax.bar(x + offset, vals, width, label=col, color=palette[i % len(palette)])

            ax.set_ylabel("Score (%)")
            ax.set_title("Architecture Comparison Across Evaluation Metrics", fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(df_metrics["Model"], fontsize=10)
            ax.set_ylim(0, 105)
            ax.legend(loc="lower right")
            ax.grid(axis="y", linestyle="--", alpha=0.5)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Architectural Analysis & Trade-offs")
        st.markdown("""
        - **ResNet18 (Residual Learning):** Residual skip connections alleviate the vanishing gradient problem, enabling the network to learn rich feature representations with high generalization across MRI textures.
        - **MobileNetV3 (Lightweight Depthwise Separable Convolutions):** Designed for high computational efficiency, achieving competitive accuracy with minimal memory footprint and fast inference.
        - **Custom CNN (Baseline):** A 4-block feedforward convolutional network providing a foundational benchmark to quantify the advantages of transfer learning.
        """)

    else:
        st.warning("`model_comparison.csv` was not found in the project root directory.")
        st.info("Place your evaluation metrics CSV in the working directory to visualize performance benchmarks.")


# =========================================================
# PAGE 4: PROJECT & DATASET INFORMATION
# =========================================================
elif nav_choice == "ℹ️ Project Information":
    st.header("Project Overview & Specifications")
    render_medical_disclaimer()

    col_info1, col_info2 = st.columns(2, gap="large")

    with col_info1:
        st.subheader("🎯 Academic Objectives")
        st.markdown("""
        - Construct an end-to-end multi-class classification pipeline for brain MRI scans.
        - Evaluate performance trade-offs across **ResNet18**, **MobileNetV3**, and a **Custom CNN**.
        - Provide explainable AI (XAI) diagnostics via **Grad-CAM** saliency maps.
        - Present results through an intuitive, reproducible web interface.
        """)

        st.subheader("📊 Dataset Details")
        st.markdown("""
        - **Source:** `masoudnickparvar/brain-tumor-mri-dataset` (Kaggle).
        - **Total Images:** 7,023 brain MRI images across 4 distinct classes.
        - **Standard Input Size:** $224 \\times 224$ pixels, 3-channel RGB.
        - **ImageNet Normalization:**
          - $\\mu = [0.485, 0.456, 0.406]$
          - $\\sigma = [0.229, 0.224, 0.225]$
        """)

    with col_info2:
        st.subheader("🔬 Clinical Class Categorization")
        st.markdown("""
        1. **Glioma:** Infiltrative primary brain tumors originating from glial cells.
        2. **Meningioma:** Typically benign extra-axial tumors originating from the meninges.
        3. **Pituitary Tumor:** Benign adenomas located in the sella turcica, impacting endocrine balance.
        4. **No Tumor (Control):** Healthy MRI scans exhibiting anatomical symmetry without mass lesions.
        """)

        st.subheader("🛠️ Technical Stack")
        st.markdown("""
        - **Deep Learning Framework:** PyTorch & Torchvision
        - **Web Application:** Streamlit
        - **Explainability:** Grad-CAM (Target Layer Gradients)
        - **Data Processing & Visualization:** NumPy, Pandas, Matplotlib, PIL
        - **Deployment Harness:** Google Colab + Tunneling (LocalTunnel / Cloudflare)
        """)

    st.markdown("---")
    st.subheader("⚠️ Limitations & Ethical Considerations")
    st.markdown("""
    - **Single-Slice 2D Evaluation:** Clinical neuroradiology evaluates volumetric 3D sequences (T1, T2, FLAIR). This 2D slice classifier lacks volumetric anatomical context.
    - **Grad-CAM Limitations:** Saliency maps highlight class-discriminative activations, but can exhibit sensitivity to high-frequency background artifacts and do not constitute segmentation boundaries.
    - **Dataset Domain Shift:** Training data originates from public retrospective datasets; performance on external institutional scanners and diverse patient demographics requires multi-center validation.
    """)
