# Trained Model Weights & Architecture Checkpoints

This directory houses the model weight checkpoints (`.pth`) for the Brain Tumor Detection system. Due to GitHub's file size limits and repository hygiene standards, large binary files are excluded from git tracking by default via `.gitignore`.

---

## 📋 Model Overview

| Model Architecture | Base Architecture | Parameters | Weight File Size | Primary Purpose | Test Accuracy | Macro F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Custom CNN** | 4-block Conv2D + BatchNorm + AdaptiveAvgPool | ~423 K | ~1.63 MB | Lightweight baseline, rapid inference | 78.75% | 0.7778 |
| **ResNet18** | Pretrained ResNet-18 (Torchvision) | ~11.2 M | ~42.72 MB | High-capacity residual feature extractor | **94.75%** | **0.9462** |
| **MobileNetV3** | Pretrained MobileNetV3 (Torchvision) | ~4.2 M | ~16.24 MB | Efficient edge-oriented classification | 93.94% | 0.9379 |

All models accept an input tensor of shape `(Batch_Size, 3, 224, 224)` normalized using standard ImageNet statistics:
- **Mean:** `[0.485, 0.456, 0.406]`
- **Std:** `[0.229, 0.224, 0.225]`

Target output classes (4 classes):
1. `glioma` (Index 0)
2. `meningioma` (Index 1)
3. `notumor` (Index 2)
4. `pituitary` (Index 3)

---

## 📥 How to Obtain Trained Model Weights

You have three options for acquiring or using the model weights:

### Option 1: Train from Scratch via Google Colab (Recommended)
1. Open [`notebooks/brain_tumor_detection.ipynb`](../notebooks/brain_tumor_detection.ipynb) in Google Colab.
2. Select a GPU runtime (**Runtime > Change runtime type > T4 GPU**).
3. Execute all cells. The notebook will automatically download the Kaggle dataset, train all three architectures for 10 epochs, evaluate metrics, and package the saved weights as `brain_tumor_results.zip`.
4. Extract the zip file and place the `.pth` files into this `models/` folder:
   ```text
   models/
   ├── custom_cnn.pth
   ├── resnet18.pth
   ├── mobilenetv3.pth
   └── README.md
   ```

### Option 2: Download from GitHub Releases / Model Hosting
If you have published the model weights as a GitHub Release:
1. Navigate to the **Releases** section of this repository.
2. Download `custom_cnn.pth`, `resnet18.pth`, and `mobilenetv3.pth`.
3. Move the downloaded `.pth` files into the `models/` folder (or project root).

### Option 3: Generate Local Dummy Weights for UI Testing
To verify the Streamlit application layout and test UI components immediately without downloading full weights:
```bash
python generate_dummy_weights.py
```
This script instantiates the architectures with initialized weights and saves them so the Streamlit app can be tested locally.

---

## 🔄 Using Git Large File Storage (Git LFS)

If you wish to version control and push the trained `.pth` files directly to GitHub:

1. Install Git LFS:
   ```powershell
   git lfs install
   ```
2. Track `.pth` binary files:
   ```powershell
   git lfs track "*.pth"
   git add .gitattributes
   ```
3. Remove `*.pth` from `.gitignore` or force-add the models:
   ```powershell
   git add -f models/*.pth
   git commit -m "Add model checkpoints via Git LFS"
   ```

