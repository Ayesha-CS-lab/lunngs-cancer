# Data Pipeline Implementation Guide

## Overview

The data pipeline and preprocessing modules have been fully implemented and are ready to use. This guide shows you how to test and use them.

## ✅ Implemented Components

### 1. **Preprocessing Module** ([src/preprocessing.py](file:///src/preprocessing.py))

**Features:**

- ✅ Load PNG, JPEG, and DICOM images
- ✅ Automatic DICOM to HU conversion
- ✅ HU windowing for CT scans (lung window: center=-600, width=1500)
- ✅ Resize to any target size
- ✅ ImageNet normalization
- ✅ Save processed images

**Code:**

```python
from src.preprocessing import load_image, normalize_image

# Load any image format
image = load_image('path/to/ct_scan.dcm', image_size=224)
# Returns: numpy array [224, 224, 3], range [0, 1]

# Normalize for deep learning
normalized = normalize_image(image)
# Returns: numpy array with ImageNet normalization
```

### 2. **Augmentation Module** ([src/augmentations.py](file:///src/augmentations.py))

**Features:**

- ✅ Albumentations-based augmentation pipeline
- ✅ Medical-appropriate transforms
- ✅ Separate train/validation transforms
- ✅ Test-Time Augmentation (TTA) support

**Code:**

```python
from src.augmentations import get_train_transforms, get_val_transforms

# For training (with random augmentation)
train_transform = get_train_transforms(image_size=224)
augmented = train_transform(image=image)
image_tensor = augmented['image']  # PyTorch tensor [3, 224, 224]

# For validation (resize + normalize only)
val_transform = get_val_transforms(image_size=224)
```

**Augmentations included:**

- Horizontal flip (50%)
- Vertical flip (20%)
- Random rotation (±15°)
- Brightness/contrast adjustment
- Gaussian blur
- Coarse dropout (cutout)

### 3. **Dataset Module** ([src/datasets.py](file:///src/datasets.py))

**Features:**

- ✅ PyTorch Dataset class
- ✅ Automatic class label detection
- ✅ K-fold stratified splitting
- ✅ Patient-level splitting (prevents leakage)
- ✅ DataLoader creation

**Code:**

```python
from src.datasets import LungCancerDataset, create_dataloaders

# Option 1: Use automatic dataloader creation
train_loader, val_loader = create_dataloaders(
    data_dir='data/processed',
    batch_size=32,
    num_workers=4,
    train_transform=get_train_transforms(224),
    val_transform=get_val_transforms(224)
)

# Option 2: Manual dataset creation
dataset = LungCancerDataset(
    image_paths=['path/to/img1.png', 'path/to/img2.png'],
    labels=[0, 1],  # 0=no cancer, 1=cancer
    transform=get_train_transforms(224)
)
```

### 4. **Utilities** ([src/utils.py](file:///src/utils.py))

**Features:**

- ✅ Seed setting for reproducibility
- ✅ Directory creation
- ✅ Model checkpointing
- ✅ HU windowing for CT scans

**Code:**

```python
from src.utils import set_seed, ensure_dir, apply_hu_windowing

# Set seeds for reproducibility
set_seed(42)

# Create directories
ensure_dir('outputs/plots')

# Apply HU windowing to DICOM images
windowed_image = apply_hu_windowing(
    dicom_image,
    center=-600,  # Lung window
    width=1500
)
```

## 🧪 Testing the Implementation

### Quick Test (Demo Script)

```bash
# 1. Install dependencies first
pip install -r requirements.txt

# 2. Run the demo
python demo_data_pipeline.py
```

**What the demo does:**

1. Creates dummy CT scan images
2. Tests image loading and preprocessing
3. Applies augmentations
4. Creates PyTorch datasets
5. Sets up DataLoaders
6. Generates visualization

**Output:**

- Console logs showing each step
- Generated images in `data/demo/`
- Visualization in `outputs/demo/pipeline_visualization.png`

### Unit Tests

```bash
# Run comprehensive unit tests
python test_data_pipeline.py
```

**What it tests:**

- Image loading (PNG/JPEG)
- Preprocessing functions
- Augmentation pipeline
- Dataset creation
- DataLoader batching
- Full integration

## 📊 Data Organization

### Required Directory Structure

```
data/
├── raw/                    # Your original CT scans
│   ├── no_cancer/
│   │   ├── patient001.png
│   │   ├── patient002.dcm
│   │   └── ...
│   └── cancer/
│       ├── patient101.png
│       ├── patient102.dcm
│       └── ...
│
└── processed/             # Auto-created by pipeline
    ├── train/
    │   ├── no_cancer/
    │   └── cancer/
    └── val/
        ├── no_cancer/
        └── cancer/
```

## 🚀 Usage Examples

### Example 1: Load and Preprocess Single Image

```python
from src.preprocessing import load_image
from src.augmentations import get_val_transforms
import cv2

# Load image
image = load_image('data/raw/cancer/sample.png', image_size=224)

# Apply validation transforms (for inference)
transform = get_val_transforms(224)
augmented = transform(image=image)
image_tensor = augmented['image']  # Ready for model

print(f"Shape: {image_tensor.shape}")  # [3, 224, 224]
print(f"Type: {image_tensor.dtype}")   # torch.float32
```

### Example 2: Create Training DataLoader

```python
from src.datasets import create_dataloaders
from src.augmentations import get_train_transforms, get_val_transforms

# Create loaders
train_loader, val_loader = create_dataloaders(
    data_dir='data/processed',
    batch_size=32,
    num_workers=4,
    train_transform=get_train_transforms(224),
    val_transform=get_val_transforms(224)
)

# Iterate through batches
for images, labels in train_loader:
    print(f"Batch images: {images.shape}")  # [32, 3, 224, 224]
    print(f"Batch labels: {labels.shape}")  # [32]
    break
```

### Example 3: Process DICOM with HU Windowing

```python
from src.preprocessing import load_image
import pydicom

# Load DICOM (automatically applies HU windowing)
image = load_image('data/raw/cancer/scan.dcm', image_size=224)

# The function automatically:
# 1. Reads DICOM file
# 2. Converts to HU values (if RescaleSlope/Intercept available)
# 3. Applies lung window (center=-600, width=1500)
# 4. Resizes to 224x224
# 5. Normalizes to [0, 1]
```

### Example 4: Create K-Fold Splits

```python
from src.datasets import create_kfold_splits

# Create 5-fold splits
splits = create_kfold_splits(
    data_dir='data/raw',
    n_folds=5,
    save_dir='data/splits',
    seed=42
)

# Splits are saved as CSV files:
# - data/splits/fold_0_train.csv
# - data/splits/fold_0_val.csv
# - ... (for all 5 folds)

print(f"Created {len(splits)} folds")
```

### Example 5: Custom Augmentation Pipeline

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Create custom augmentation
custom_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Use with dataset
from src.datasets import LungCancerDataset

dataset = LungCancerDataset(
    image_paths=your_paths,
    labels=your_labels,
    transform=custom_transform  # Use custom transform
)
```

## 🔧 Configuration

All settings are in `configs/config.py`:

```python
# Image settings
IMAGE_SIZE = 224           # Input size for models
BATCH_SIZE = 32            # Batch size for training
NUM_WORKERS = 4            # DataLoader workers

# HU Windowing (for DICOM)
HU_WINDOW_CENTER = -600    # Lung window center
HU_WINDOW_WIDTH = 1500     # Lung window width

# Reproducibility
SEED = 42                  # Random seed
```

## ✅ Verification Checklist

Run these checks to verify your pipeline:

```bash
# 1. Check dependencies
python -c "import numpy, cv2, torch, albumentations; print('✓ All imports work')"

# 2. Test image loading
python -c "from src.preprocessing import load_image; print('✓ Preprocessing module works')"

# 3. Test augmentations
python -c "from src.augmentations import get_train_transforms; print('✓ Augmentation module works')"

# 4. Test dataset
python -c "from src.datasets import LungCancerDataset; print('✓ Dataset module works')"

# 5. Run full demo
python demo_data_pipeline.py

# 6. Run unit tests
python test_data_pipeline.py
```

## 🎯 Next Steps

Once the pipeline is working:

1. **Add your data**: Place CT scans in `data/raw/no_cancer/` and `data/raw/cancer/`
2. **Verify data**: Run `python demo_data_pipeline.py` to check loading
3. **Train model**: Run `python train.py --model efficientnet`
4. **Monitor**: Check `outputs/` for results

## 📝 Technical Details

### Image Format Support

| Format | Extension       | HU Windowing | Notes                               |
| ------ | --------------- | ------------ | ----------------------------------- |
| PNG    | `.png`          | No           | Standard image format               |
| JPEG   | `.jpg`, `.jpeg` | No           | Compressed format                   |
| DICOM  | `.dcm`          | Yes          | Medical standard, includes metadata |

### Preprocessing Pipeline

```
Raw Image → Load → [HU Windowing (if DICOM)] → Resize → Normalize → Ready
```

### Augmentation Pipeline

**Training:**

```
Image → Resize → Flip → Rotate → Brightness → Blur → Dropout → Normalize → Tensor
```

**Validation:**

```
Image → Resize → Normalize → Tensor
```

## 🐛 Troubleshooting

**Problem: `ModuleNotFoundError`**

```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**Problem: DICOM not loading**

```bash
# Solution: Install GDCM
pip install gdcm
```

**Problem: Out of memory**

```python
# Solution: Reduce batch size in configs/config.py
BATCH_SIZE = 16  # Instead of 32
NUM_WORKERS = 2  # Instead of 4
```

**Problem: Images too large**

```python
# Solution: Reduce image size
IMAGE_SIZE = 128  # Instead of 224
```

## 📚 Module Reference

### preprocessing.py

- `load_image(path, image_size)` - Load and preprocess any image
- `normalize_image(image, mean, std)` - Apply ImageNet normalization
- `save_processed_image(image, path)` - Save preprocessed image

### augmentations.py

- `get_train_transforms(image_size)` - Get training augmentations
- `get_val_transforms(image_size)` - Get validation transforms
- `get_tta_transforms(image_size)` - Get TTA transforms

### datasets.py

- `LungCancerDataset` - PyTorch Dataset class
- `create_dataloaders(...)` - Create train/val loaders
- `create_kfold_splits(...)` - Create K-fold splits

### utils.py

- `set_seed(seed)` - Set random seeds
- `ensure_dir(path)` - Create directory
- `apply_hu_windowing(image, center, width)` - HU windowing for CT

---

**The data pipeline is fully implemented and ready to use!** 🎉
