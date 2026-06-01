# Installation Guide - Troubleshooting Network Issues

## Problem: Network Timeout During Installation

If you're experiencing timeout errors during `pip install -r requirements.txt`, here are solutions:

---

## ✅ Solution 1: Install Minimal Dependencies (Recommended)

For **dataset preparation only**, you only need 3 packages:

```bash
pip install --default-timeout=100 kagglehub
pip install --default-timeout=100 tqdm
pip install --default-timeout=100 Pillow
```

Or run the batch script:

```bash
install_minimal.bat
```

Then you can prepare the dataset:

```bash
python prepare_kaggle_dataset.py
```

---

## ✅ Solution 2: Install in Stages

Install heavy packages separately:

### Stage 1: Core ML Libraries (Heavy - ~2GB)

```bash
pip install --default-timeout=100 torch torchvision
```

⏱️ This takes 10-30 minutes

### Stage 2: Image Processing

```bash
pip install --default-timeout=100 opencv-python Pillow
```

### Stage 3: Scientific Computing

```bash
pip install --default-timeout=100 numpy scikit-learn matplotlib
```

### Stage 4: Augmentation & Utilities

```bash
pip install --default-timeout=100 albumentations tqdm
```

### Stage 5: Additional Tools

```bash
pip install --default-timeout=100 pydicom xgboost joblib gradio kagglehub pandas seaborn
```

---

## ✅ Solution 3: Use Pre-built Wheels

If you have slow internet, download wheels manually:

1. Go to https://download.pytorch.org/whl/torch_stable.html
2. Download:
   - `torch-2.0.0+cpu-cp39-cp39-win_amd64.whl`
   - `torchvision-0.15.0+cpu-cp39-cp39-win_amd64.whl`
3. Install locally:

```bash
pip install torch-2.0.0+cpu-cp39-cp39-win_amd64.whl
pip install torchvision-0.15.0+cpu-cp39-cp39-win_amd64.whl
```

---

## ✅ Solution 4: Increase Timeout

```bash
pip install --default-timeout=200 -r requirements.txt
```

---

## ✅ Solution 5: Use Conda (If Available)

```bash
conda install pytorch torchvision -c pytorch
conda install opencv scikit-learn matplotlib
pip install albumentations gradio kagglehub
```

---

## 🎯 What You Need for Each Task

### Just Dataset Preparation:

```bash
pip install kagglehub tqdm Pillow
```

### Training Models:

```bash
pip install torch torchvision opencv-python numpy scikit-learn albumentations tqdm
```

### Full Project (including demos):

```bash
pip install -r requirements.txt
```

---

## 🐛 Common Errors

### Error: "Read timed out"

**Solution**: Increase timeout or retry

```bash
pip install --default-timeout=200 <package>
```

### Error: "No module named 'X'"

**Solution**: Install missing package

```bash
pip install <package-name>
```

### Error: "CUDA not available"

**Solution**: Install CPU version

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## ✅ Verification

After installation, verify:

```bash
python setup_check.py
```

Or manually check:

```python
import torch
import cv2
import kagglehub
print("✓ All dependencies installed!")
```

---

## 🚀 Quick Start After Minimal Install

Once you have `kagglehub`, `tqdm`, and `Pillow`:

```bash
# 1. Prepare dataset
python prepare_kaggle_dataset.py

# 2. Install remaining dependencies later when needed
pip install torch torchvision opencv-python numpy scikit-learn
```

---

**Recommended**: Install minimal dependencies first, prep data, then install full dependencies overnight or when you have better internet.
