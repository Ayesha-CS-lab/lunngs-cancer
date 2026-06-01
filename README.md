# 🫁 Lung Cancer Detection AI System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-quality **Explainable Stacked Ensemble Model** for lung cancer detection from CT scan images using PyTorch,featuring:

✅ **Transfer Learning** with EfficientNet, DenseNet, and ResNet  
✅ **GAN-based Data Augmentation** for minority class  
✅ **Stacked Ensemble Learning** with meta-learners  
✅ **Explainable AI** using Grad-CAM visualization  
✅ **Comprehensive Evaluation** with calibration curves  
✅ **Interactive Gradio Demo** for deployment

---

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Usage](#-usage)
  - [Training Base Models](#1-training-base-models)
  - [GAN Augmentation](#2-gan-augmentation-optional)
  - [Ensemble Training](#3-ensemble-training)
  - [Evaluation](#4-evaluation)
  - [Demo Application](#5-demo-application)
- [Configuration](#-configuration)
- [Medical ML Best Practices](#-medical-ml-best-practices)
- [Results](#-results)
- [Citations](#-citations)

---

## 🚀 Features

### 🧠 Deep Learning Models

- **Transfer Learning**: Pretrained EfficientNet, DenseNet, and ResNet
- **Two-Phase Training**: Frozen backbone → Fine-tuning
- **Mixed Precision**: Automatic mixed precision (AMP) for faster training
- **Class Imbalance**: Weighted loss functions

### 🎨 Data Augmentation

- **Albumentations**: Advanced augmentation pipeline
- **GAN-based**: Conditional GAN for synthetic minority samples
- **DICOM Support**: HU windowing for CT scans

### 📊 Ensemble Learning

- **Stacked Ensemble**: K-fold out-of-fold predictions
- **Meta-Learners**: Logistic Regression, Random Forest, XGBoost
- **Prevents Leakage**: Stratified patient-level splitting

### 🔍 Explainability

- **Grad-CAM**: Class activation mapping for each model
- **Heatmap Overlay**: Visual explanations for predictions
- **Ensemble Heatmaps**: Averaged explanations across models

### 📈 Evaluation

- Accuracy, Precision, Recall, Specificity, F1-Score
- ROC AUC with curves
- Confusion Matrix
- Calibration Curves

---

## 📁 Project Structure

```
lung_cancer_ai/
│
├── data/
│   ├── raw/              # Original CT scan images
│   ├── processed/        # Preprocessed images
│   │   ├── train/
│   │   │   ├── no_cancer/
│   │   │   └── cancer/
│   │   └── val/
│   │       ├── no_cancer/
│   │       └── cancer/
│   └── splits/           # K-fold split metadata
│
├── src/
│   ├── datasets.py       # PyTorch Dataset & DataLoaders
│   ├── preprocessing.py  # Image preprocessing (DICOM, HU windowing)
│   ├── augmentations.py  # Albumentations transforms
│   │
│   ├── gan/
│   │   ├── generator.py      # Conditional GAN generator
│   │   ├── discriminator.py  # Conditional GAN discriminator
│   │   ├── train_gan.py      # GAN training loop
│   │   └── sample.py         # Synthetic image generation
│   │
│   ├── models/
│   │   ├── base_models.py  # EfficientNet, DenseNet, ResNet
│   │   ├── trainer.py      # Training loop with AMP
│   │   └── inference.py    # Prediction & TTA
│   │
│   ├── ensemble/
│   │   ├── stacking.py      # Stacked ensemble logic
│   │   └── meta_models.py   # Meta-learner models
│   │
│   ├── explainability/
│   │   └── gradcam.py      # Grad-CAM implementation
│   │
│   ├── evaluation.py   # Metrics & visualization
│   └── utils.py        # Utilities (seed, checkpointing, etc.)
│
├── configs/
│   └── config.py       # Hyperparameters & paths
│
├── checkpoints/        # Model weights
├── outputs/            # Evaluation plots & heatmaps
├── experiments/        # Experiment logs
│
├── train.py            # Main training script
├── evaluate.py         # Evaluation script
├── demo_app.py         # Gradio demo interface
│
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

---

## 🔧 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup

```bash
# Clone repository (or navigate to project directory)
cd lung_cancer_ai

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 Data Preparation

### 1. Organize Your Data

Place your CT scan images in the following structure:

```
data/raw/
├── no_cancer/
│   ├── patient1_scan1.png
│   ├── patient2_scan1.png
│   └── ...
└── cancer/
    ├── patient3_scan1.png
    ├── patient4_scan1.png
    └── ...
```

**Supported formats**: `.png`, `.jpg`, `.jpeg`, `.dcm` (DICOM)

### 2. Preprocess Data

The data pipeline automatically:

- Loads DICOM or PNG/JPEG images
- Applies HU windowing (for DICOM CT scans)
- Resizes to 224×224
- Normalizes using ImageNet statistics
- Creates train/val splits (patient-level stratified)

---

## 🎯 Usage

### 1. Training Base Models

Train individual CNN models:

```bash
# Train EfficientNet
python train.py --model efficientnet --epochs 50 --batch-size 32

# Train DenseNet
python train.py --model densenet --epochs 50 --batch-size 32

# Train ResNet
python train.py --model resnet --epochs 50 --batch-size 32
```

**Training Process:**

- **Phase 1**: Train with frozen backbone (transfer learning)
- **Phase 2**: Fine-tune with unfrozen backbone (lower LR)
- Automatic early stopping
- Mixed precision training (AMP)
- Class-weighted loss

### 2. GAN Augmentation (Optional)

Train GAN to generate synthetic minority-class samples:

```bash
# Train GAN
python -m src.gan.train_gan

# Generate synthetic images
python -m src.gan.sample
```

**Configuration**: Edit `configs/config.py`:

```python
GAN_ENABLED = True
GAN_EPOCHS = 100
GAN_LATENT_DIM = 100
```

### 3. Ensemble Training

Train stacked ensemble with K-fold cross-validation:

```python
from src.ensemble import StackedEnsemble
from configs.config import *

# Create ensemble
ensemble = StackedEnsemble(
    base_models=['efficientnet', 'densenet', 'resnet'],
    n_folds=5,
    device=DEVICE
)

# Generate out-of-fold predictions
oof_preds, oof_labels = ensemble.generate_oof_predictions(
    train_loader_fn,  # Function returning train loader for fold
    val_loader_fn,     # Function returning val loader for fold
    num_epochs=50
)

# Train meta-learner
ensemble.train_meta_model(oof_preds, oof_labels, meta_learner='xgboost')
```

### 4. Evaluation

Evaluate trained models:

```bash
python evaluate.py \
    --model efficientnet \
    --checkpoint checkpoints/efficientnet_finetuned_best.pth \
    --data-dir data/processed
```

**Outputs:**

- Confusion matrix
- ROC curve
- Calibration curve
- Comprehensive metrics report

### 5. Demo Application

Launch interactive Gradio demo:

```bash
python demo_app.py
```

**Features:**

- Upload CT scan image
- Select model (EfficientNet, DenseNet, ResNet)
- View prediction with confidence
- Toggle Grad-CAM explainability heatmap

---

## ⚙️ Configuration

Edit `configs/config.py` to customize:

```python
# Data
IMAGE_SIZE = 224
BATCH_SIZE = 32

# Training
EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 10  # Early stopping

# GAN
GAN_ENABLED = True
GAN_EPOCHS = 100

# Ensemble
N_FOLDS = 5
ENSEMBLE_MODELS = ["efficientnet", "densenet", "resnet"]
META_LEARNER = "xgboost"  # or "logistic", "random_forest"

# Class weights for imbalance
CLASS_WEIGHTS = [1.0, 2.0]  # [No Cancer, Cancer]
```

---

## 🏥 Medical ML Best Practices

This project follows medical imaging best practices:

✅ **Patient-Level Splitting**: Prevents data leakage  
✅ **Stratified Sampling**: Maintains class distribution  
✅ **HU Windowing**: Proper CT scan preprocessing  
✅ **Calibration**: Ensures probability estimates are reliable  
✅ **Explainability**: Grad-CAM for clinical interpretability  
✅ **Class Imbalance**: Weighted loss functions  
✅ **Reproducibility**: Seeded random states

⚠️ **Disclaimer**: This is a research/educational tool. Not FDA-approved. Not for clinical use without validation.

---

## 📈 Results

### Expected Performance

| Metric      | Value      |
| ----------- | ---------- |
| Accuracy    | ~85-92%    |
| Precision   | ~80-88%    |
| Recall      | ~82-90%    |
| Specificity | ~85-93%    |
| F1-Score    | ~81-89%    |
| ROC AUC     | ~0.90-0.95 |

_Results depend on dataset quality and size_

### Explainability Example

![Grad-CAM Example](outputs/heatmaps/example_gradcam.png)

---

## 🔬 How It Works

### 1. Stacked Ensemble Architecture

```
CT Scan Image
     │
     ├─→ EfficientNet ──→ Probabilities [0.2, 0.8]
     ├─→ DenseNet     ──→ Probabilities [0.3, 0.7]  ──→ Meta-Features
     └─→ ResNet       ──→ Probabilities [0.25, 0.75]
                                                          │
                                                          ▼
                                                    Meta-Learner
                                                     (XGBoost)
                                                          │
                                                          ▼
                                                  Final Prediction
```

### 2. Out-of-Fold Predictions

- Train each base model on K-fold splits
- Generate predictions on held-out folds
- Stack predictions as meta-features
- Train meta-learner on OOF predictions
- **Prevents overfitting** and data leakage

### 3. Grad-CAM Explainability

- Computes gradients of target class w.r.t. feature maps
- Weighted combination of activation maps
- Highlights regions influencing prediction
- **Critical for clinical trust**

---

## 📝 Code Examples

### Single Image Prediction

```python
from src.models import get_model
from src.models.inference import predict_single_image
from src.augmentations import get_val_transforms
import cv2

# Load model
model = get_model('efficientnet', pretrained=False)
model.load_state_dict(torch.load('checkpoints/efficientnet_best.pth'))

# Load image
image = cv2.imread('scan.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Predict
transform = get_val_transforms()
pred_class, confidence, probs = predict_single_image(model, image, transform)

print(f"Prediction: {'Cancer' if pred_class == 1 else 'No Cancer'}")
print(f"Confidence: {confidence:.2%}")
```

### Generate Grad-CAM

```python
from src.explainability.gradcam import visualize_gradcam

overlaid, heatmap, pred = visualize_gradcam(
    model,
    'efficientnet',
    image,
    image_tensor,
    save_path='outputs/heatmaps/gradcam.png'
)
```

---

## 🛠️ Troubleshooting

### CUDA Out of Memory

- Reduce `BATCH_SIZE` in `configs/config.py`
- Use smaller image size (e.g., 128 instead of 224)
- Disable mixed precision: `use_amp=False`

### Poor Performance

- Increase `EPOCHS`
- Adjust `CLASS_WEIGHTS` for severe imbalance
- Enable GAN augmentation
- Use Test-Time Augmentation (TTA)

### DICOM Loading Issues

- Install: `pip install gdcm`
- Verify DICOM metadata (RescaleSlope, RescaleIntercept)

---

## 📚 Citations

If you use this code, please consider citing:

```bibtex
@software{lung_cancer_ai,
  title={Explainable Stacked Ensemble for Lung Cancer Detection},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/lung_cancer_ai}
}
```

### Key References

- **Grad-CAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks" (ICCV 2017)
- **EfficientNet**: Tan & Le, "EfficientNet: Rethinking Model Scaling" (ICML 2019)
- **Transfer Learning**: Shin et al., "Deep Convolutional Neural Networks for Medical Image Analysis" (MIA 2016)

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## ⚠️ Important Notes

### Data Privacy

- **Do NOT** commit patient data to version control
- Use anonymized datasets only
- Comply with HIPAA/GDPR regulations

### Clinical Use

- This tool is for **research and education only**
- Not FDA-approved or clinically validated
- Always consult healthcare professionals

---

## 📧 Contact

For questions or issues, please open a GitHub issue or contact [your-email@example.com]

---

## 🙏 Acknowledgments

- PyTorch team for the framework
- Albumentations for augmentation library
- Gradio for demo interface tools
- Open-source medical imaging community

---

**Made with ❤️ for advancing medical AI**
