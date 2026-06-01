# Quick Start Guide

## Installation

1. **Install Python 3.8+**

2. **Create virtual environment:**

   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify setup:**
   ```bash
   python setup_check.py
   ```

## Prepare Your Data

1. **Organize CT scans:**

   ```
   data/raw/
   ├── no_cancer/
   │   ├── scan001.png
   │   └── ...
   └── cancer/
       ├── scan101.png
       └── ...
   ```

2. **Supported formats:** PNG, JPEG, DICOM (.dcm)

## Training Workflow

### Option 1: Train Individual Model

```bash
python train.py --model efficientnet --epochs 50
```

### Option 2: Train All Models

```bash
# EfficientNet
python train.py --model efficientnet --epochs 50

# DenseNet
python train.py --model densenet --epochs 50

# ResNet
python train.py --model resnet --epochs 50
```

### Option 3: Enable GAN Augmentation

1. Edit `configs/config.py`:

   ```python
   GAN_ENABLED = True
   ```

2. Train GAN:

   ```bash
   python -m src.gan.train_gan
   ```

3. Generate synthetic images:
   ```bash
   python -m src.gan.sample
   ```

## Evaluation

```bash
python evaluate.py \
    --model efficientnet \
    --checkpoint checkpoints/efficientnet_finetuned_best.pth
```

## Launch Demo

```bash
python demo_app.py
```

Then open the URL shown in terminal (usually `http://127.0.0.1:7860`)

## Tips

- **Out of memory?** Reduce `BATCH_SIZE` in `configs/config.py`
- **Slow training?** Use smaller `IMAGE_SIZE` (e.g., 128)
- **Poor performance?** Enable GAN augmentation or adjust `CLASS_WEIGHTS`

## Troubleshooting

### CUDA not found

- Install PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

### Import errors

- Run: `pip install -r requirements.txt --upgrade`

### DICOM loading fails

- Install: `pip install gdcm`

For more details, see [README.md](README.md)
