"""
Model Architectures and Preprocessing Pipelines for Brain Tumor Detection.
Supports Custom CNN, ResNet18, and MobileNetV3 with robust state_dict loading.
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


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


class CustomCNN(nn.Module):
    """
    Standard 4-block Convolutional Neural Network designed for 224x224 MRI images.
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
    """Instantiate a base model architecture configured for num_classes."""
    name = model_name.strip().lower()

    if "resnet18" in name:
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    elif "mobilenetv3" in name or "mobilenet" in name:
        # Default to MobileNetV3-Large
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model

    elif "custom" in name or "cnn" in name:
        return CustomCNN(num_classes=num_classes)

    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")


def clean_state_dict(checkpoint):
    """Extract raw state_dict and remove DataParallel prefixes if present."""
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        # Look for common container keys
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
        # Strip DataParallel or DistributedDataParallel prefixes
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
        
        # If checkpoint is an entire pickled model
        if isinstance(checkpoint, nn.Module):
            model = checkpoint
            model.to(device)
            model.eval()
            return model, "Model loaded successfully as serialized nn.Module."

        cleaned_dict = clean_state_dict(checkpoint)

        # Auto-align MobileNetV3 small vs large variant if needed
        if isinstance(model, models.MobileNetV3):
            has_large_keys = any("features.15" in k for k in cleaned_dict.keys())
            is_large_model = len(model.features) > 13
            if has_large_keys and not is_large_model:
                num_classes = model.classifier[3].out_features
                model = models.mobilenet_v3_large(weights=None)
                model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
            elif not has_large_keys and is_large_model:
                num_classes = model.classifier[3].out_features
                model = models.mobilenet_v3_small(weights=None)
                model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

        # Try strict load first
        try:
            model.load_state_dict(cleaned_dict, strict=True)
            model.to(device)
            model.eval()
            return model, "Weights loaded successfully (strict match)."
        except RuntimeError as strict_err:
            # Fallback to non-strict loading if minor head or buffer mismatch
            missing, unexpected = model.load_state_dict(cleaned_dict, strict=False)
            model.to(device)
            model.eval()
            return model, (
                f"Weights loaded with non-strict match. Missing keys: {len(missing)}, "
                f"Unexpected keys: {len(unexpected)}."
            )

    except Exception as e:
        return model, f"Error loading weights: {str(e)}"

