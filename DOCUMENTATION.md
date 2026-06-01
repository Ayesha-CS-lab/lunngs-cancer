# 📚 Lung Cancer Detection AI - Complete Documentation

## 📖 Table of Contents

1. [Introduction](#introduction)
2. [System Overview](#system-overview)
3. [How It Works - Step by Step](#how-it-works---step-by-step)
4. [Installation & Setup](#installation--setup)
5. [Data Preparation](#data-preparation)
6. [Training Process](#training-process)
7. [Evaluation & Testing](#evaluation--testing)
8. [Using the Demo Application](#using-the-demo-application)
9. [Understanding the Code](#understanding-the-code)
10. [Advanced Features](#advanced-features)
11. [Troubleshooting](#troubleshooting)

---

## 1. Introduction

### What is this project?

This is an **AI-powered medical imaging system** that analyzes CT scans to detect lung cancer. It uses:

- **Deep Learning** (CNN models) to learn from images
- **Ensemble Learning** to combine multiple models for better accuracy
- **Explainable AI** to show which parts of the image influenced the decision

### Who is it for?

- **Researchers** studying medical AI
- **Students** learning deep learning
- **Developers** building healthcare applications
- **Data Scientists** working with medical imaging

### Key Capabilities

✅ Detects lung cancer from CT scans with high accuracy  
✅ Explains predictions with visual heatmaps (Grad-CAM)  
✅ Handles class imbalance with GAN-based augmentation  
✅ Provides confidence scores for clinical decision support

---

## 2. System Overview

### The Big Picture

```
Your CT Scan → AI System → Prediction + Explanation
                ↓
        [Cancer: 85% confidence]
        [Heatmap showing tumor region]
```

### What happens inside?

1. **Data Processing**: CT scan is preprocessed (resized, normalized)
2. **Model Inference**: Three different AI models analyze the image
3. **Ensemble Combination**: Predictions are combined intelligently
4. **Explainability**: System generates heatmap showing important regions
5. **Output**: You get prediction + confidence + visual explanation

### Architecture Components

| Component          | Purpose                      | Technology                     |
| ------------------ | ---------------------------- | ------------------------------ |
| **Data Pipeline**  | Load and preprocess CT scans | PyTorch, OpenCV, pydicom       |
| **Base Models**    | Analyze images               | EfficientNet, DenseNet, ResNet |
| **Ensemble**       | Combine predictions          | XGBoost meta-learner           |
| **Explainability** | Generate heatmaps            | Grad-CAM                       |
| **Demo App**       | User interface               | Gradio                         |

---

## 3. How It Works - Step by Step

### Step 1: Data Preprocessing

**What happens:**

```
Raw CT Scan (DICOM/PNG)
    ↓
HU Windowing (for DICOM) - Adjusts brightness for lung tissue
    ↓
Resize to 224×224 - Standard input size
    ↓
Normalization - Scales pixel values
    ↓
Ready for AI Model
```

**Why it matters:**

- Medical images need special preprocessing (HU windowing)
- Standardized size allows batch processing
- Normalization helps models learn faster

**Code example:**

```python
from src.preprocessing import load_image

# Automatically handles DICOM or PNG/JPEG
image = load_image('path/to/ct_scan.dcm', image_size=224)
```

---

### Step 2: Data Augmentation (Optional GAN)

**What happens:**

If you have **class imbalance** (e.g., 70% no-cancer, 30% cancer):

```
Real Cancer Images (300)
    ↓
Train Conditional GAN (100 epochs)
    ↓
Generate Synthetic Cancer Images (500)
    ↓
Balanced Dataset (800 cancer, 800 no-cancer)
```

**Why it matters:**

- Models trained on imbalanced data perform poorly
- GAN generates realistic synthetic images
- Balances the dataset without collecting more real data

**How to enable:**

```python
# In configs/config.py
GAN_ENABLED = True

# Then train GAN
python -m src.gan.train_gan

# Generate synthetic images
python -m src.gan.sample
```

---

### Step 3: Base Model Training

**What happens:**

We train **three different CNN models**:

```
CT Scan Image
    ↓
┌─────────────┬─────────────┬─────────────┐
│ EfficientNet│  DenseNet   │   ResNet    │
│  (Model 1)  │  (Model 2)  │  (Model 3)  │
└─────────────┴─────────────┴─────────────┘
    ↓              ↓              ↓
[Prediction 1] [Prediction 2] [Prediction 3]
```

**Two-Phase Training Process:**

**Phase 1: Transfer Learning (25 epochs)**

```
Pretrained Weights (from ImageNet)
    ↓
Freeze the backbone (feature extractor)
    ↓
Train only the classification head
    ↓
Fast initial learning
```

**Phase 2: Fine-Tuning (25 epochs)**

```
Unfreeze the backbone
    ↓
Reduce learning rate (10x smaller)
    ↓
Fine-tune all layers
    ↓
Adapt to medical images
```

**Why three models?**

- Different architectures learn different patterns
- EfficientNet: Best efficiency
- DenseNet: Rich feature connections
- ResNet: Stable training

**Training command:**

```bash
# Train one model
python train.py --model efficientnet --epochs 50

# Or train all three
python train.py --model efficientnet --epochs 50
python train.py --model densenet --epochs 50
python train.py --model resnet --epochs 50
```

---

### Step 4: Stacked Ensemble Learning

**What happens:**

Instead of averaging predictions, we use a **meta-learner**:

```
Training Phase (5-Fold Cross-Validation):
──────────────────────────────────────────

Fold 1: Train on folds 2-5, predict on fold 1
Fold 2: Train on folds 1,3-5, predict on fold 2
Fold 3: Train on folds 1-2,4-5, predict on fold 3
Fold 4: Train on folds 1-3,5, predict on fold 4
Fold 5: Train on folds 1-4, predict on fold 5

Result: Out-of-Fold (OOF) predictions for entire dataset
        (no data leakage!)

These OOF predictions become features for meta-learner:
──────────────────────────────────────────

For each image, we have:
┌───────────────────────────────────────┐
│ EfficientNet: [0.2, 0.8] (no, yes)  │
│ DenseNet:     [0.3, 0.7]             │
│ ResNet:       [0.25, 0.75]           │
└───────────────────────────────────────┘
        ↓
Meta-Features: [0.2, 0.8, 0.3, 0.7, 0.25, 0.75]
        ↓
XGBoost Meta-Learner learns to combine them
        ↓
Final Prediction: 82% confident it's cancer
```

**Why this is better than averaging:**

- Meta-learner learns which model to trust more
- Captures complex interactions between predictions
- Typically 2-5% better than best single model

---

### Step 5: Explainability with Grad-CAM

**What happens:**

Grad-CAM shows **where the model is looking**:

```
1. Forward Pass:
   CT Scan → Model → Prediction: "Cancer"

2. Backward Pass:
   Compute gradients of "Cancer" w.r.t. feature maps
   ↓
   "Which pixels contributed most to this prediction?"

3. Generate Heatmap:
   Weighted combination of feature maps
   ↓
   Normalize to [0, 1]
   ↓
   Apply color map (red = high importance)

4. Overlay:
   Heatmap + Original Image
   ↓
   Visual explanation
```

**Example output:**

```
Original Image        Heatmap           Overlay
┌─────────┐          ┌─────────┐       ┌─────────┐
│  Lung   │    →     │ ███🔴██ │   →   │ Lung🔴  │
│ Tissue  │          │ ████▓██ │       │ with    │
│         │          │ ████▓██ │       │ tumor   │
└─────────┘          └─────────┘       └─────────┘
                     High activation    highlighted
                     on tumor region
```

**Why it matters:**

- **Clinical Trust**: Doctors need to understand AI decisions
- **Validation**: Ensures model looks at correct regions
- **Debugging**: Detects if model uses wrong features

**How to use:**

```python
from src.explainability.gradcam import visualize_gradcam

# Generate Grad-CAM
overlaid, heatmap, pred_class = visualize_gradcam(
    model=my_model,
    model_name='efficientnet',
    image=original_image,
    image_tensor=preprocessed_tensor,
    save_path='output/gradcam.png'
)
```

---

## 4. Installation & Setup

### Prerequisites

- **Python 3.8+** installed
- **GPU recommended** (NVIDIA with CUDA) but CPU works too
- **16GB RAM** minimum

### Step-by-Step Installation

**1. Navigate to project directory:**

```bash
cd C:\Users\pc\.gemini\antigravity\scratch\lung_cancer_ai
```

**2. Create virtual environment:**

```bash
python -m venv venv
```

**3. Activate virtual environment:**

```bash
# Windows:
venv\Scripts\activate

# You should see (venv) in your terminal
```

**4. Install dependencies:**

```bash
pip install -r requirements.txt
```

This installs:

- PyTorch (deep learning)
- torchvision (image processing)
- efficientnet-pytorch (EfficientNet model)
- albumentations (augmentation)
- scikit-learn (evaluation)
- xgboost (meta-learner)
- gradio (demo interface)
- And more...

**5. Verify installation:**

```bash
python setup_check.py
```

You should see:

```
✓ torch
✓ torchvision
✓ efficientnet_pytorch
...
✅ All dependencies installed!
✅ CUDA available! (or "⚠️ CPU only")
✅ Directory structure created!
```

---

## 5. Data Preparation

### Organizing Your Data

**Required structure:**

```
data/raw/
├── no_cancer/
│   ├── patient001_scan.png
│   ├── patient002_scan.png
│   ├── patient003_scan.dcm
│   └── ...
└── cancer/
    ├── patient101_scan.png
    ├── patient102_scan.png
    ├── patient103_scan.dcm
    └── ...
```

### File Format Support

| Format | Extension       | Notes                                |
| ------ | --------------- | ------------------------------------ |
| PNG    | `.png`          | Most common, easy to use             |
| JPEG   | `.jpg`, `.jpeg` | Compressed format                    |
| DICOM  | `.dcm`          | Medical standard, includes HU values |

### Data Split

The system automatically creates train/validation splits:

```
Original Data (1000 images)
    ↓
Stratified Split (maintains class ratio)
    ↓
Training: 800 images (80%)
Validation: 200 images (20%)
```

**Patient-level splitting** ensures:

- Same patient's scans don't appear in both train and test
- Prevents data leakage
- Realistic performance estimates

---

## 6. Training Process

### Simple Training (One Model)

**Train a single model:**

```bash
python train.py --model efficientnet --epochs 50
```

**What happens:**

1. Loads and preprocesses data
2. Creates data loaders
3. Initializes EfficientNet model
4. **Phase 1**: Trains with frozen backbone (25 epochs)
5. **Phase 2**: Fine-tunes with unfrozen backbone (25 epochs)
6. Saves best model checkpoint
7. Generates evaluation plots

**Expected output:**

```
======================================================================
Training EFFICIENTNET Model
======================================================================

Device: cuda
Creating data loaders...
Train batches: 25
Val batches: 6

Creating efficientnet model...
Trainable parameters: 4,234,123

======================================================================
PHASE 1: Training with Frozen Backbone
======================================================================

Epoch 1/25
Training: 100%|████████████| 25/25 [00:45<00:00]
Train Loss: 0.5234 | Train Acc: 76.50%
Val Loss: 0.4123 | Val Acc: 82.00%
✓ New best model saved!

...

======================================================================
PHASE 2: Fine-tuning with Unfrozen Backbone
======================================================================

Learning rate reduced to 0.00001

Epoch 26/50
...

✓ Training completed!
✓ Best model saved to: checkpoints/efficientnet_finetuned_best.pth
```

### Training All Models (Prepare for Ensemble)

```bash
# Train all three base models
python train.py --model efficientnet --epochs 50
python train.py --model densenet --epochs 50
python train.py --model resnet --epochs 50
```

### Configuration Options

Edit `configs/config.py` to customize:

```python
# Training
EPOCHS = 50              # Total epochs (split into 2 phases)
LEARNING_RATE = 1e-4     # Initial learning rate
BATCH_SIZE = 32          # Images per batch

# Data
IMAGE_SIZE = 224         # Input size (224x224)
NUM_CLASSES = 2          # Binary classification

# Class imbalance
CLASS_WEIGHTS = [1.0, 2.0]  # [No Cancer, Cancer]
                             # Higher weight = more importance

# GAN
GAN_ENABLED = True       # Enable GAN augmentation
GAN_EPOCHS = 100         # GAN training epochs
```

### Training Tips

**For limited GPU memory:**

```python
BATCH_SIZE = 16  # Reduce from 32
IMAGE_SIZE = 128 # Reduce from 224
```

**For faster training:**

```python
EPOCHS = 30      # Reduce epochs
use_amp = True   # Enable mixed precision (already default)
```

**For better accuracy:**

```python
CLASS_WEIGHTS = [1.0, 3.0]  # Increase minority class weight
GAN_ENABLED = True           # Enable synthetic data
```

---

## 7. Evaluation & Testing

### Evaluate a Trained Model

```bash
python evaluate.py \
    --model efficientnet \
    --checkpoint checkpoints/efficientnet_finetuned_best.pth \
    --data-dir data/processed
```

### Evaluation Outputs

**1. Console Metrics:**

```
============================================================
EVALUATION REPORT
============================================================

Accuracy:     0.9150 (91.50%)
Precision:    0.8823
Recall:       0.8947
Specificity:  0.9350
F1-Score:     0.8885
ROC AUC:      0.9456

------------------------------------------------------------
Classification Report:
------------------------------------------------------------

              precision    recall  f1-score   support

   No Cancer       0.94      0.94      0.94       120
      Cancer       0.88      0.89      0.89        80

    accuracy                           0.92       200
   macro avg       0.91      0.92      0.91       200
weighted avg       0.92      0.92      0.92       200
```

**2. Visual Plots (saved in `outputs/plots/`):**

- **Confusion Matrix**: Shows TP, TN, FP, FN
- **ROC Curve**: Discrimination ability (AUC score)
- **Calibration Curve**: Probability reliability

### Understanding Metrics

| Metric                   | What it means                            | Good value       |
| ------------------------ | ---------------------------------------- | ---------------- |
| **Accuracy**             | Overall correctness                      | >90%             |
| **Precision**            | Of predicted cancers, how many are real? | >85%             |
| **Recall (Sensitivity)** | Of real cancers, how many did we catch?  | >90% (critical!) |
| **Specificity**          | Of healthy patients, how many correct?   | >85%             |
| **F1-Score**             | Balance of precision & recall            | >85%             |
| **ROC AUC**              | Overall discrimination ability           | >0.90            |

**For medical AI, Recall (Sensitivity) is MOST important:**

- High recall = fewer missed cancers
- False negative (missing cancer) is worse than false positive

---

## 8. Using the Demo Application

### Launch the Demo

```bash
python demo_app.py
```

**Output:**

```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxxx.gradio.live (shareable link)
```

### Demo Interface Features

**1. Upload Image:**

- Drag & drop CT scan
- Or click to browse files
- Supports PNG, JPEG, DICOM

**2. Select Model:**

- Choose EfficientNet, DenseNet, or ResNet
- Different models may have different strengths

**3. Toggle Grad-CAM:**

- ✅ Enabled: Shows heatmap overlay
- ❌ Disabled: Shows original image only

**4. Click "Analyze":**

- Processing takes 2-5 seconds
- Results appear on right side

### Understanding Results

**Prediction Output:**

```
## Prediction: Cancer

**Confidence:** 85.23%

**Class Probabilities:**
- No Cancer: 14.77%
- Cancer: 85.23%
```

**Visual Output:**

- Original image with red/yellow heatmap overlay
- Red areas = regions influencing "Cancer" prediction
- Should highlight tumor location

### Demo Use Cases

**1. Quick Testing:**
Upload single scans to test model performance

**2. Clinical Demonstration:**
Show healthcare professionals how AI works

**3. Educational:**
Help students understand deep learning

**4. Debugging:**
Visual check if model looks at correct regions

---

## 9. Understanding the Code

### Project Structure Explained

```
lung_cancer_ai/
│
├── configs/
│   └── config.py              # All settings in one place
│
├── src/
│   ├── datasets.py            # Data loading (PyTorch Dataset)
│   ├── preprocessing.py       # Image preprocessing
│   ├── augmentations.py       # Image augmentation
│   ├── utils.py               # Helper functions
│   ├── evaluation.py          # Metrics and plots
│   │
│   ├── models/
│   │   ├── base_models.py     # CNN architectures
│   │   ├── trainer.py         # Training loop
│   │   └── inference.py       # Prediction
│   │
│   ├── gan/
│   │   ├── generator.py       # GAN generator
│   │   ├── discriminator.py   # GAN discriminator
│   │   ├── train_gan.py       # GAN training
│   │   └── sample.py          # Generate synthetic images
│   │
│   ├── ensemble/
│   │   ├── stacking.py        # K-fold stacking
│   │   └── meta_models.py     # Meta-learners
│   │
│   └── explainability/
│       └── gradcam.py         # Grad-CAM implementation
│
├── train.py                   # Main training script
├── evaluate.py                # Evaluation script
├── demo_app.py                # Gradio demo
├── setup_check.py             # Verify installation
│
├── requirements.txt           # Python dependencies
├── README.md                  # Project overview
└── QUICKSTART.md             # Quick guide
```

### Key Files Explained

**[configs/config.py](file:///C:/Users/pc/.gemini/antigravity/scratch/lung_cancer_ai/configs/config.py)**

```python
# Central configuration - change settings here
EPOCHS = 50          # How many training cycles
BATCH_SIZE = 32      # Images per batch
LEARNING_RATE = 1e-4 # Learning speed
```

**[src/models/base_models.py](file:///C:/Users/pc/.gemini/antigravity/scratch/lung_cancer_ai/src/models/base_models.py)**

```python
# Defines CNN architectures
class EfficientNetModel(nn.Module):
    # EfficientNet implementation

class DenseNetModel(nn.Module):
    # DenseNet implementation

class ResNetModel(nn.Module):
    # ResNet implementation
```

**[src/models/trainer.py](file:///C:/Users/pc/.gemini/antigravity/scratch/lung_cancer_ai/src/models/trainer.py)**

```python
# Training loop with:
# - Mixed precision (AMP)
# - Class weights
# - Early stopping
# - Learning rate scheduling
```

**[src/explainability/gradcam.py](file:///C:/Users/pc/.gemini/antigravity/scratch/lung_cancer_ai/src/explainability/gradcam.py)**

```python
# Grad-CAM implementation
# Generates visual explanations
```

---

## 10. Advanced Features

### Feature 1: GAN-based Data Augmentation

**When to use:**

- Severe class imbalance (e.g., 80-20 split)
- Limited minority class samples
- Want to improve recall

**How to enable:**

```python
# 1. Edit configs/config.py
GAN_ENABLED = True
GAN_EPOCHS = 100

# 2. Train GAN
python -m src.gan.train_gan

# 3. Generate synthetic images
python -m src.gan.sample --num_samples 500 --target_class 1

# 4. Synthetic images saved to data/synthetic/
# 5. Include them in training by copying to data/raw/cancer/
```

### Feature 2: Test-Time Augmentation (TTA)

**What is it:**
Make predictions on multiple augmented versions, then average:

```python
from src.models.inference import predict_with_tta
from src.augmentations import get_tta_transforms

# Predict with TTA (more robust)
pred_class, confidence, probs = predict_with_tta(
    model=my_model,
    image=my_image,
    tta_transforms=get_tta_transforms(),
    device='cuda'
)
```

**Benefits:**

- 1-2% accuracy improvement
- More robust to image orientation
- Useful for edge cases

### Feature 3: Ensemble Stacking

**Full ensemble training:**

```python
from src.ensemble import StackedEnsemble

# Create ensemble
ensemble = StackedEnsemble(
    base_models=['efficientnet', 'densenet', 'resnet'],
    n_folds=5,
    device='cuda'
)

# Train all models with K-fold
oof_preds, oof_labels = ensemble.generate_oof_predictions(
    train_loader_fn=lambda fold: get_train_loader(fold),
    val_loader_fn=lambda fold: get_val_loader(fold),
    num_epochs=50
)

# Train meta-learner
ensemble.train_meta_model(oof_preds, oof_labels, meta_learner='xgboost')

# Predict on new data
predictions, probabilities = ensemble.predict_ensemble(test_loader)
```

### Feature 4: Custom Callbacks

**Add TensorBoard logging:**

```python
# In trainer.py, add:
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs/experiment1')

# During training:
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Accuracy/val', val_acc, epoch)

# View with:
# tensorboard --logdir=logs
```

---

## 11. Troubleshooting

### Common Issues & Solutions

**1. CUDA Out of Memory**

```
RuntimeError: CUDA out of memory
```

**Solution:**

```python
# In configs/config.py
BATCH_SIZE = 16  # Reduce from 32
IMAGE_SIZE = 128 # Reduce from 224
```

**2. DICOM Loading Error**

```
AttributeError: 'FileDataset' object has no attribute 'RescaleSlope'
```

**Solution:**

```bash
pip install gdcm
# Or convert DICOM to PNG first
```

**3. Low Accuracy**

```
Validation accuracy stuck at ~60%
```

**Possible causes:**

- Insufficient data → Use GAN augmentation
- Class imbalance → Increase `CLASS_WEIGHTS`
- Too short training → Increase `EPOCHS`
- Wrong normalization → Check preprocessing

**4. Grad-CAM Not Highlighting Tumor**
**Possible causes:**

- Model not well-trained → Train longer
- Wrong target layer → Check `get_target_layer()`
- Model using wrong features → Check training data quality

**5. Import Errors**

```
ModuleNotFoundError: No module named 'efficientnet_pytorch'
```

**Solution:**

```bash
pip install -r requirements.txt --upgrade
```

---

## 📝 Quick Reference

### Common Commands

```bash
# Setup
python setup_check.py

# Train single model
python train.py --model efficientnet --epochs 50

# Evaluate
python evaluate.py --model efficientnet --checkpoint checkpoints/efficientnet_best.pth

# Demo
python demo_app.py

# GAN augmentation
python -m src.gan.train_gan
python -m src.gan.sample
```

### File Locations

- **Checkpoints**: `checkpoints/`
- **Evaluation plots**: `outputs/plots/`
- **Grad-CAM heatmaps**: `outputs/heatmaps/`
- **Training data**: `data/processed/`
- **Configuration**: `configs/config.py`

---

## 🎓 Learning Path

**For Beginners:**

1. Read this documentation
2. Run `setup_check.py`
3. Use pre-trained demo (`demo_app.py`)
4. Train single model
5. Understand evaluation metrics

**For Intermediate:**

1. Train all three models
2. Experiment with configurations
3. Try GAN augmentation
4. Implement ensemble stacking
5. Customize Grad-CAM visualization

**For Advanced:**

1. Modify model architectures
2. Add new meta-learners
3. Implement 3D CNN for volumetric CT
4. Add attention mechanisms
5. Deploy to cloud (AWS/Azure)

---

## 🏥 Medical AI Best Practices

✅ **Patient-Level Splitting** - Never mix same patient in train/test  
✅ **Stratified Sampling** - Maintain class distribution  
✅ **Clinical Validation** - Test on external datasets  
✅ **Explainability** - Always provide visual explanations  
✅ **Calibration** - Ensure probability estimates are reliable  
✅ **Regulatory Compliance** - Follow FDA/CE guidelines for medical devices

⚠️ **Important:** This is a research/educational tool. Not for clinical use without proper validation and regulatory approval.

---

## 📞 Need Help?

1. **Check** [README.md](file:///C:/Users/pc/.gemini/antigravity/scratch/lung_cancer_ai/README.md) - Project overview
2. **Check** [QUICKSTART.md](file:///C:/Users/pc/.gemini/antigravity/scratch/lung_cancer_ai/QUICKSTART.md) - Fast setup guide
3. **Check** this documentation - Detailed explanations
4. **Review** code comments - Inline documentation
5. **Open** GitHub issue - For bugs/questions

---

**This project demonstrates production-quality medical AI with explainability!** 🏥✨
