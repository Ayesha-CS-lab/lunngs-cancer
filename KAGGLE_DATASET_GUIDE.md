# Kaggle Dataset Integration Guide

## Overview

This guide explains how to integrate the **IQ-OTHNCCD Lung Cancer Dataset** from Kaggle into your lung cancer detection project.

---

## 📦 Dataset Information

**Kaggle Dataset**: [IQ-OTHNCCD Lung Cancer Dataset](https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset)

**Original Structure:**

```
dataset/
├── Normal/      # Healthy lung tissue
├── Benign/      # Non-cancerous abnormalities
└── Malignant/   # Cancerous tumors
```

**Our Binary Classification:**

```
Reorganized as:
├── no_cancer/   # Normal + Benign
└── cancer/      # Malignant
```

This focuses on detecting **malignant cancer** specifically, which is the most clinically critical.

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install kagglehub
pip install kagglehub

# Configure Kaggle API credentials
# 1. Go to https://www.kaggle.com/account
# 2. Create new API token
# 3. Download kaggle.json
# 4. Place in:
#    - Linux/Mac: ~/.kaggle/kaggle.json
#    - Windows: C:\Users\<username>\.kaggle\kaggle.json
```

### Run Integration Script

```bash
python prepare_kaggle_dataset.py
```

**What it does:**

1. ✅ Downloads dataset from Kaggle (~500MB-1GB)
2. ✅ Analyzes class distribution
3. ✅ Reorganizes into binary classification
4. ✅ Splits into train (70%), val (15%), test (15%)
5. ✅ Prepares for model training

**Expected output:**

```
======================================================================
KAGGLE LUNG CANCER DATASET INTEGRATION
======================================================================

STEP 1: Downloading Kaggle Dataset
Downloading IQ-OTHNCCD Lung Cancer Dataset...
✓ Dataset downloaded to: /path/to/dataset

STEP 2: Analyzing Dataset Structure
✓ Normal      :  300 images
✓ Benign      :  200 images
✓ Malignant   :  500 images

Total         : 1000 images

STEP 3: Reorganizing for Binary Classification
Copying Normal images to no_cancer...
Copying Benign images to no_cancer...
Copying Malignant images to cancer...

Binary classification distribution:
  no_cancer :  500 images (Normal + Benign)
  cancer    :  500 images (Malignant)

STEP 4: Splitting into Train/Val/Test Sets
Processing no_cancer...
Processing cancer...

TRAIN :  700 images (no_cancer:  350, cancer:  350)
VAL   :  150 images (no_cancer:   75, cancer:   75)
TEST  :  150 images (no_cancer:   75, cancer:   75)

✓ Dataset split complete!

STEP 5: Creating Convenient Access Link
✓ Symbolic link created!

✅ DATASET INTEGRATION COMPLETE!
```

---

## 📊 Dataset After Integration

### Directory Structure

```
data/kaggle_lung_cancer/
├── raw/
│   ├── no_cancer/
│   │   ├── normal_img001.png
│   │   ├── normal_img002.png
│   │   ├── benign_img001.png
│   │   └── ...
│   └── cancer/
│       ├── malignant_img001.png
│       ├── malignant_img002.png
│       └── ...
├── train/
│   ├── no_cancer/  (70% of data)
│   └── cancer/
├── val/
│   ├── no_cancer/  (15% of data)
│   └── cancer/
└── test/
    ├── no_cancer/  (15% of data)
    └── cancer/
```

### Access Path

Data is accessible at:

- **Direct**: `data/kaggle_lung_cancer/`
- **Symbolic link**: `data/processed/` → `data/kaggle_lung_cancer/`

---

## 🎯 Training with Kaggle Dataset

### Train Single Model

```bash
python train.py \
    --model efficientnet \
    --epochs 50 \
    --batch_size 32 \
    --data_dir data/kaggle_lung_cancer
```

### Train All Models

```bash
# EfficientNet
python train.py --model efficientnet --epochs 50

# DenseNet
python train.py --model densenet --epochs 50

# ResNet
python train.py --model resnet --epochs 50
```

### Train with GAN Augmentation

```bash
# Step 1: Train GAN
python -m src.gan.train_gan \
    --data_dir data/kaggle_lung_cancer/train \
    --epochs 100

# Step 2: Generate synthetic images
python -m src.gan.sample \
    --num_samples 500 \
    --target_class 1 \
    --output_dir data/kaggle_lung_cancer/train/cancer

# Step 3: Train models
python train.py --model efficientnet --epochs 50
```

---

## ⚙️ Customization Options

### Change Split Ratios

Edit `prepare_kaggle_dataset.py`:

```python
# Default: 70% train, 15% val, 15% test
split_dataset(
    target_path,
    train_ratio=0.8,   # 80% train
    val_ratio=0.1,     # 10% val
    test_ratio=0.1,    # 10% test
    seed=42
)
```

### Alternative Classification Strategy

Option 1 (Current): **Detect Malignant Cancer**

```python
# Normal + Benign → no_cancer
# Malignant → cancer
```

Option 2: **Detect Any Abnormality**

```python
# In prepare_kaggle_dataset.py, reorganize_for_binary_classification():

# Copy Normal to no_cancer
# Copy Benign + Malignant to cancer

# This detects any abnormality (benign or malignant)
```

Option 3: **3-Class Classification**

```python
# Keep original structure
# Update configs/config.py:
NUM_CLASSES = 3
CLASS_NAMES = ['Normal', 'Benign', 'Malignant']
```

### Set Custom Output Path

```python
# In prepare_kaggle_dataset.py, main():

target_path = reorganize_for_binary_classification(
    dataset_path,
    target_base="data/my_custom_path"  # Custom path
)
```

---

## 🔍 Data Quality Checks

### Verify Dataset

```python
from pathlib import Path

data_path = Path("data/kaggle_lung_cancer")

for split in ["train", "val", "test"]:
    for class_name in ["no_cancer", "cancer"]:
        path = data_path / split / class_name
        count = len(list(path.glob("*")))
        print(f"{split}/{class_name}: {count} images")
```

### Check Image Properties

```python
import cv2
from pathlib import Path

# Sample 10 images
images = list(Path("data/kaggle_lung_cancer/train/cancer").glob("*"))[:10]

for img_path in images:
    img = cv2.imread(str(img_path))
    print(f"{img_path.name}: Shape={img.shape}, Dtype={img.dtype}")
```

### Visualize Samples

```python
import matplotlib.pyplot as plt
from src.preprocessing import load_image

# Load samples
no_cancer_imgs = list(Path("data/kaggle_lung_cancer/train/no_cancer").glob("*"))[:5]
cancer_imgs = list(Path("data/kaggle_lung_cancer/train/cancer").glob("*"))[:5]

fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i, img_path in enumerate(no_cancer_imgs):
    img = load_image(str(img_path), image_size=224)
    axes[0, i].imshow(img)
    axes[0, i].set_title("No Cancer")
    axes[0, i].axis('off')

for i, img_path in enumerate(cancer_imgs):
    img = load_image(str(img_path), image_size=224)
    axes[1, i].imshow(img)
    axes[1, i].set_title("Cancer")
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig("outputs/dataset_samples.png")
plt.show()
```

---

## 📈 Expected Performance

### With This Dataset

| Model        | Expected Accuracy | Training Time (50 epochs) |
| ------------ | ----------------- | ------------------------- |
| EfficientNet | 88-92%            | ~30-40 min (GPU)          |
| DenseNet     | 86-90%            | ~45-60 min (GPU)          |
| ResNet       | 87-91%            | ~50-70 min (GPU)          |
| Ensemble     | 90-94%            | N/A (combine models)      |

### Class Imbalance Handling

If class imbalance is detected:

```python
# Automatically handled in configs/config.py
# Adjust CLASS_WEIGHTS based on actual distribution

# Example: If ratio is 2:1 (no_cancer:cancer)
CLASS_WEIGHTS = [1.0, 2.0]  # Give cancer 2x importance
```

---

## 🐛 Troubleshooting

### Problem: Kaggle API Not Configured

**Error:**

```
OSError: Could not find kaggle.json
```

**Solution:**

1. Go to https://www.kaggle.com/account
2. Click "Create New API Token"
3. Save `kaggle.json` to `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Problem: Download Fails

**Error:**

```
403 Forbidden
```

**Solution:**

1. Accept dataset rules on Kaggle website
2. Ensure API token is valid
3. Try manual download:
   ```bash
   kaggle datasets download -d adityamahimkar/iqothnccd-lung-cancer-dataset
   unzip iqothnccd-lung-cancer-dataset.zip -d data/kaggle_download
   ```

### Problem: Disk Space Insufficient

**Error:**

```
OSError: No space left on device
```

**Solution:**

```bash
# Check available space
df -h

# Clean up old data
rm -rf data/demo_*  # Remove demo datasets

# Or use external drive
python prepare_kaggle_dataset.py
# Edit target_base="/path/to/external/drive"
```

### Problem: Symbolic Link Failed (Windows)

**Error:**

```
OSError: symbolic link privilege not held
```

**Solution:**

1. **Option 1**: Run as Administrator
2. **Option 2**: Enable Developer Mode (Windows 10+)
3. **Option 3**: Use direct path:
   ```bash
   python train.py --data_dir data/kaggle_lung_cancer
   ```

---

## 📝 Manual Integration (Alternative)

If the script fails, you can integrate manually:

### Step 1: Download Dataset

```bash
kaggle datasets download -d adityamahimkar/iqothnccd-lung-cancer-dataset
unzip iqothnccd-lung-cancer-dataset.zip -d data/kaggle_raw
```

### Step 2: Reorganize

```bash
# Create structure
mkdir -p data/processed/train/no_cancer
mkdir -p data/processed/train/cancer
mkdir -p data/processed/val/no_cancer
mkdir -p data/processed/val/cancer
mkdir -p data/processed/test/no_cancer
mkdir -p data/processed/test/cancer

# Copy files (example for Normal)
cp data/kaggle_raw/Normal/* data/processed/train/no_cancer/
# Repeat for Benign → no_cancer and Malignant → cancer
```

### Step 3: Split Manually

Use a script or tool to split 70-15-15.

---

## 🎯 Best Practices

1. **Always verify data** after integration
2. **Check class balance** before training
3. **Use stratified splitting** (automatically done)
4. **Validate with test set** only once
5. **Document any preprocessing** steps
6. **Back up original dataset** before modifications

---

## 📞 Next Steps After Integration

1. ✅ **Verify data integrity**

   ```bash
   python -c "from pathlib import Path; print(len(list(Path('data/kaggle_lung_cancer/train/cancer').glob('*'))))"
   ```

2. ✅ **Train baseline model**

   ```bash
   python train.py --model efficientnet --epochs 10  # Quick test
   ```

3. ✅ **Evaluate on validation set**

   ```bash
   python evaluate.py --model efficientnet
   ```

4. ✅ **Full training**

   ```bash
   python train.py --model efficientnet --epochs 50
   ```

5. ✅ **Launch demo app**
   ```bash
   python demo_app.py
   ```

---

**The Kaggle dataset is ready for training!** 🎯✨
