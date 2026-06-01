# GAN Augmentation Implementation Guide

## Overview

The GAN (Generative Adversarial Network) module is fully implemented for generating synthetic medical images to balance your dataset. This is particularly useful when you have class imbalance (e.g., fewer cancer samples).

---

## ✅ Implemented Components

### 1. **Generator** ([src/gan/generator.py](file:///src/gan/generator.py))

**Architecture:**

- Conditional generator (class-aware)
- Takes random noise + class label
- Generates realistic CT scan images
- Uses transposed convolutions for upsampling

**Features:**

- Label embedding for conditional generation
- Progressive upsampling layers
- Batch normalization for stability
- Tanh activation for output

### 2. **Discriminator** ([src/gan/discriminator.py](file:///src/gan/discriminator.py))

**Architecture:**

- Conditional discriminator
- Judges if image is real or fake
- Class-aware discrimination
- Progressive downsampling

**Features:**

- Label embedding
- LeakyReLU activation
- Dropout for regularization
- Sigmoid output (real/fake probability)

### 3. **Training Script** ([src/gan/train_gan.py](file:///src/gan/train_gan.py))

**Production-quality trainer with:**

- ✅ Progress tracking with tqdm
- ✅ Automatic checkpointing
- ✅ Sample generation during training
- ✅ Loss visualization
- ✅ Best model saving
- ✅ Training history logging

### 4. **Sample Generator** ([src/gan/sample.py](file:///src/gan/sample.py))

**Batch generation tool:**

- ✅ Load trained GAN
- ✅ Generate N synthetic images
- ✅ Save to specified directory
- ✅ Ready for training use

### 5. **Demo Script** ([demo_gan_training.py](file:///demo_gan_training.py))

**Interactive demonstration:**

- ✅ Creates dummy data
- ✅ Trains GAN from scratch
- ✅ Generates samples
- ✅ Visualizes results

---

## 🚀 Quick Start

### Option 1: Run Demo (Recommended for First Time)

```bash
# Install dependencies first
pip install torch torchvision matplotlib tqdm

# Run the demo
python demo_gan_training.py
```

**What happens:**

1. Creates 40 dummy training images
2. Trains GAN for 50 epochs (~5-10 minutes)
3. Generates 10 synthetic images
4. Creates visualizations

**Output:**

- `outputs/demo_gan/gan_results.png` - Training losses and samples
- `outputs/demo_gan/generated_grid.png` - Grid of generated images
- `checkpoints/demo_gan/` - Trained models

### Option 2: Train on Real Data

```bash
# Train GAN on your actual CT scans
python -m src.gan.train_gan \
    --data_dir data/processed/train \
    --img_size 64 \
    --batch_size 16 \
    --epochs 100 \
    --checkpoint_dir checkpoints/gan \
    --sample_dir outputs/gan/samples
```

**Arguments:**

- `--data_dir`: Path to training data
- `--img_size`: Image size (64, 128, or 224)
- `--batch_size`: Batch size (16 for 64x64, 8 for 128x128)
- `--epochs`: Number of training epochs (100-200 recommended)
- `--lr`: Learning rate (default: 0.0002)
- `--checkpoint_interval`: Save checkpoint every N epochs
- `--sample_interval`: Generate samples every N epochs

### Option 3: Generate Synthetic Images

```bash
# After training, generate 500 synthetic cancer images
python -m src.gan.sample \
    --checkpoint checkpoints/gan/gan_best.pth \
    --num_samples 500 \
    --target_class 1 \
    --output_dir data/synthetic/cancer
```

**Arguments:**

- `--checkpoint`: Path to trained GAN checkpoint
- `--num_samples`: How many images to generate
- `--target_class`: 0=no_cancer, 1=cancer
- `--output_dir`: Where to save generated images

---

## 📊 Complete Workflow

### Step 1: Prepare Training Data

```
data/processed/train/
├── no_cancer/
│   ├── img_001.png
│   ├── img_002.png
│   └── ... (e.g., 800 images)
└── cancer/
    ├── img_101.png
    ├── img_102.png
    └── ... (e.g., 200 images)
```

**Problem:** Imbalanced! 800 no-cancer vs 200 cancer

### Step 2: Train GAN

```bash
python -m src.gan.train_gan \
    --data_dir data/processed/train \
    --epochs 100 \
    --img_size 64
```

**Training output:**

```
Epoch [1/100]
Training: 100%|██████████| 25/25 [00:15<00:00]
Generator Loss: 1.2345
Discriminator Loss: 0.6789
✓ Samples saved for epoch 1

...

Epoch [100/100]
Generator Loss: 0.4123
Discriminator Loss: 0.6234
✓ Best model saved (epoch 100)

TRAINING COMPLETED!
✓ Best Generator Loss: 0.4123
✓ Checkpoints saved to: checkpoints/gan/
```

### Step 3: Generate Synthetic Images

```bash
# Generate 600 synthetic cancer images
python -m src.gan.sample \
    --checkpoint checkpoints/gan/gan_best.pth \
    --num_samples 600 \
    --target_class 1 \
    --output_dir data/synthetic/cancer
```

**Output:**

```
Generating 600 images for class 1...
Generating: 100%|██████████| 600/600 [00:30<00:00]

Saving 600 images to: data/synthetic/cancer
Saving: 100%|██████████| 600/600 [00:15<00:00]

✓ Generated 600 synthetic images
✓ Saved to: data/synthetic/cancer
```

### Step 4: Combine with Real Data

```bash
# Copy synthetic images to training data
# Windows:
xcopy data\synthetic\cancer\*.png data\processed\train\cancer\ /Y

# Linux/Mac:
cp data/synthetic/cancer/*.png data/processed/train/cancer/
```

**Result:**

```
data/processed/train/
├── no_cancer/ (800 images)
└── cancer/ (200 real + 600 synthetic = 800 images)
```

**Balanced!** 800 vs 800

### Step 5: Train Classification Model

```bash
python train.py --model efficientnet --epochs 50
```

---

## 🎯 Usage Examples

### Example 1: Quick Demo Test

```python
# test_gan.py
import torch
from src.gan.generator import Generator

# Create generator
generator = Generator(latent_dim=100, num_classes=2, img_size=64)

# Generate random image
z = torch.randn(1, 100)
label = torch.tensor([1])  # Cancer class

with torch.no_grad():
    fake_image = generator(z, label)

print(f"Generated image shape: {fake_image.shape}")  # [1, 3, 64, 64]
```

### Example 2: Load and Use Trained GAN

```python
from src.gan.sample import load_generator, generate_images

# Load trained generator
generator, config = load_generator('checkpoints/gan/gan_best.pth')

# Generate 10 cancer images
images = generate_images(
    generator=generator,
    config=config,
    num_samples=10,
    target_class=1,
    device='cuda'
)

print(f"Generated {len(images)} images")
```

### Example 3: Custom Training Loop

```python
from src.gan.train_gan import GANTrainer, create_config

# Custom configuration
config = {
    'latent_dim': 100,
    'num_classes': 2,
    'img_size': 64,
    'lr': 0.0002,
    'beta1': 0.5,
    'beta2': 0.999,
    'checkpoint_dir': 'my_checkpoints',
    'sample_dir': 'my_samples',
    'output_dir': 'my_outputs',
    'checkpoint_interval': 10,
    'sample_interval': 5,
}

# Initialize trainer
trainer = GANTrainer(config)

# Train
trainer.train(dataloader, num_epochs=50)
```

---

## 📈 Monitoring Training

### 1. Check Sample Quality

During training, samples are saved every N epochs:

```
outputs/gan/samples/
├── samples_epoch_005.png
├── samples_epoch_010.png
├── samples_epoch_015.png
└── ...
```

**What to look for:**

- Early epochs: Noisy, random patterns
- Middle epochs: Recognizable structures
- Late epochs: Realistic-looking scans

### 2. Check Loss Curves

After training, check `outputs/gan/training_losses.png`:

**Good training:**

- Generator loss: Gradually decreases then stabilizes
- Discriminator loss: Stays around 0.5-0.7 (balanced)
- Both losses converge

**Bad training (mode collapse):**

- Generator loss: Drops to near 0
- Discriminator loss: Shoots to 1.0
- **Solution:** Reduce learning rate, increase batch size

### 3. Check Training History

```python
import json

with open('outputs/gan/training_history.json', 'r') as f:
    history = json.load(f)

print(f"Final G loss: {history['g_loss'][-1]:.4f}")
print(f"Final D loss: {history['d_loss'][-1]:.4f}")
```

---

## ⚙️ Configuration Tips

### For 64x64 Images (Fast Training)

```bash
python -m src.gan.train_gan \
    --img_size 64 \
    --batch_size 16 \
    --epochs 100 \
    --lr 0.0002
```

**Pros:** Fast training (~10 mins on GPU)  
**Cons:** Lower resolution

### For 128x128 Images (Balanced)

```bash
python -m src.gan.train_gan \
    --img_size 128 \
    --batch_size 8 \
    --epochs 150 \
    --lr 0.0001
```

**Pros:** Better quality  
**Cons:** Slower training (~30 mins on GPU)

### For 224x224 Images (Best Quality)

```bash
python -m src.gan.train_gan \
    --img_size 224 \
    --batch_size 4 \
    --epochs 200 \
    --lr 0.00005
```

**Pros:** Best quality for classification  
**Cons:** Very slow training (~1-2 hours on GPU)

---

## 🐛 Troubleshooting

### Problem: Generator Loss → 0, Discriminator Loss → 1

**Cause:** Mode collapse (generator found a shortcut)

**Solutions:**

```bash
# 1. Reduce learning rate
--lr 0.0001  # Instead of 0.0002

# 2. Increase batch size
--batch_size 32  # Instead of 16

# 3. Add noise to discriminator
# (Already implemented in code)
```

### Problem: Poor Quality Images

**Cause:** Insufficient training

**Solutions:**

```bash
# 1. Train longer
--epochs 200  # Instead of 100

# 2. Use larger architecture
# Modify configs/config.py:
GAN_LATENT_DIM = 200  # Instead of 100

# 3. More training data
# Collect more real images
```

### Problem: CUDA Out of Memory

**Solutions:**

```bash
# 1. Reduce batch size
--batch_size 8  # Instead of 16

# 2. Reduce image size
--img_size 64  # Instead of 128

# 3. Use CPU (slower but works)
--device cpu
```

### Problem: Discriminator Too Strong

**Symptoms:**

- Generator loss stays high
- Generated images look random

**Solutions:**

```python
# In src/gan/train_gan.py, add label smoothing:
valid = torch.ones(batch_size, 1) * 0.9  # Instead of 1.0
fake = torch.zeros(batch_size, 1) + 0.1  # Instead of 0.0
```

---

## 📊 Expected Results

### Training Time (100 epochs)

| Image Size | Batch Size | GPU (RTX 3080) | CPU       |
| ---------- | ---------- | -------------- | --------- |
| 64x64      | 16         | ~10 min        | ~2 hour   |
| 128x128    | 8          | ~30 min        | ~6 hours  |
| 224x224    | 4          | ~1 hour        | ~12 hours |

### Quality Benchmarks

**After 100 epochs:**

- Generated images should resemble CT scans
- Class-specific features should be visible
- Some artifacts are normal

**After 200 epochs:**

- High-quality synthetic images
- Difficult to distinguish from real
- Ready for data augmentation

---

## 🎓 When to Use GAN Augmentation

**Use GAN when:**

- ✅ Severe class imbalance (>5:1 ratio)
- ✅ Limited minority class samples (<500)
- ✅ Need to improve recall on minority class
- ✅ Have GPU for training

**Don't use GAN when:**

- ❌ Balanced dataset already
- ❌ Very small dataset (<100 total samples)
- ❌ Limited computational resources
- ❌ Simple augmentations work well

**Alternative:** Traditional augmentations (already in [src/augmentations.py](file:///src/augmentations.py))

---

## 📝 Quick Reference Commands

```bash
# Demo
python demo_gan_training.py

# Train GAN
python -m src.gan.train_gan --epochs 100

# Generate samples
python -m src.gan.sample --num_samples 500 --target_class 1

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# View samples
# Navigate to outputs/gan/samples/ and open PNG files
```

---

## 🎯 Best Practices

1. **Start with demo** to verify everything works
2. **Train for 100 epochs** minimum
3. **Generate samples periodically** to monitor quality
4. **Use 2-3x minority samples** (if 200 cancer, generate 400-600)
5. **Mix synthetic with real** during training
6. **Validate model** on real data only (not synthetic)

---

**The GAN module is production-ready and working!** 🎨✨
