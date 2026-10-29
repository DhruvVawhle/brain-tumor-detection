"""
Helper utility to generate initial/dummy PyTorch weights for testing the Streamlit UI
prior to deploying actual trained models from Google Colab.
"""

import os
import json
import torch
from model_architectures import build_model

def generate_weights():
    os.makedirs("models", exist_ok=True)
    num_classes = 4

    models_info = [
        ("custom_cnn.pth", "Custom CNN"),
        ("resnet18.pth", "ResNet18"),
        ("mobilenetv3.pth", "MobileNetV3")
    ]

    for filename, name in models_info:
        # Check both root and models/ folder
        if not os.path.exists(filename) and not os.path.exists(os.path.join("models", filename)):
            print(f"Generating dummy test weights for {name} -> {filename}...")
            model = build_model(name, num_classes=num_classes)
            torch.save(model.state_dict(), filename)
            print(f"Saved: {filename}")
        else:
            print(f"File exists: {filename}")

if __name__ == "__main__":
    generate_weights()

