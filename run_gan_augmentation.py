"""
GAN Augmentation Pipeline
=========================
Trains a Conditional GAN on real CT scan data, then generates synthetic
images to balance the dataset before CNN training.

Run this BEFORE train.py so the base models train on augmented data.

Usage:
    python run_gan_augmentation.py
    python run_gan_augmentation.py --epochs 200 --extra-samples 100
    python run_gan_augmentation.py --skip-training  # if GAN already trained
"""
import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from configs.config import *
from src.gan.train_gan import GANTrainer
from src.gan.sample import load_generator, generate_images
from src.utils import set_seed, ensure_dir

# GAN trains at 64x64 (fast, memory-efficient on Kaggle T4).
# Generated images are then resized to IMAGE_SIZE (224x224) before saving.
GAN_IMG_SIZE = 64
TRAIN_DIR    = "data/kaggle_lung_cancer/train"


# ---------------------------------------------------------------------------
# Dataset that loads real CT images resized to GAN_IMG_SIZE for GAN training
# ---------------------------------------------------------------------------
class GANTrainDataset(Dataset):
    def __init__(self, data_dir, img_size=64):
        self.image_paths = []
        self.labels = []
        data_dir = Path(data_dir)

        for class_idx, class_name in enumerate(["no_cancer", "cancer"]):
            class_dir = data_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.glob("*")):
                if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    # Skip any previously generated synthetic images
                    if img_path.name.startswith("synthetic_"):
                        continue
                    self.image_paths.append(str(img_path))
                    self.labels.append(class_idx)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def count_real_images(data_dir):
    """Count real (non-synthetic) images per class."""
    counts = {}
    for class_name in ["no_cancer", "cancer"]:
        class_dir = Path(data_dir) / class_name
        if class_dir.exists():
            counts[class_name] = sum(
                1 for p in class_dir.glob("*")
                if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
                and not p.name.startswith("synthetic_")
            )
        else:
            counts[class_name] = 0
    return counts


def remove_previous_synthetic(data_dir):
    """Delete synthetic images from a previous run to start fresh."""
    removed = 0
    for class_name in ["no_cancer", "cancer"]:
        class_dir = Path(data_dir) / class_name
        if class_dir.exists():
            for img_path in class_dir.glob("synthetic_*.png"):
                img_path.unlink()
                removed += 1
    if removed:
        print(f"  Removed {removed} synthetic images from previous run.")


def save_synthetic_images(raw_images, output_dir, target_size=224):
    """Resize GAN output (64x64) to target_size and save as PNG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, img in enumerate(raw_images):
        resized = cv2.resize(img, (target_size, target_size),
                             interpolation=cv2.INTER_CUBIC)
        save_path = output_dir / f"synthetic_{i:04d}.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
        saved += 1
    return saved


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GAN-based data augmentation pipeline")
    parser.add_argument("--data-dir",       type=str, default=TRAIN_DIR)
    parser.add_argument("--epochs",         type=int, default=150,
                        help="GAN training epochs (150+ recommended)")
    parser.add_argument("--batch-size",     type=int, default=16)
    parser.add_argument("--extra-samples",  type=int, default=50,
                        help="Extra synthetic images beyond class balance")
    parser.add_argument("--skip-training",  action="store_true",
                        help="Skip GAN training; load existing checkpoint instead")
    parser.add_argument("--gan-checkpoint", type=str,
                        default="checkpoints/gan/gan_best.pth")
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("GAN AUGMENTATION PIPELINE")
    print("=" * 70)
    print(f"Device : {device}")
    print(f"Data   : {args.data_dir}")
    print(f"Epochs : {args.epochs}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Analyse class distribution
    # ------------------------------------------------------------------
    print("Step 1: Analysing class distribution...")
    counts = count_real_images(args.data_dir)
    no_cancer_n = counts["no_cancer"]
    cancer_n    = counts["cancer"]
    print(f"  Real images — no_cancer: {no_cancer_n}, cancer: {cancer_n}")

    minority_class = "cancer" if cancer_n <= no_cancer_n else "no_cancer"
    minority_idx   = 1 if minority_class == "cancer" else 0
    minority_n     = min(cancer_n, no_cancer_n)
    majority_n     = max(cancer_n, no_cancer_n)
    n_to_generate  = (majority_n - minority_n) + args.extra_samples

    print(f"  Minority class : {minority_class} ({minority_n} images)")
    print(f"  Will generate  : {n_to_generate} synthetic '{minority_class}' images "
          f"(balance gap {majority_n - minority_n} + {args.extra_samples} extra)")

    # Remove stale synthetic images from previous runs
    print("\n  Cleaning up previous synthetic images...")
    remove_previous_synthetic(args.data_dir)

    # ------------------------------------------------------------------
    # Step 2: Train GAN (or load existing checkpoint)
    # ------------------------------------------------------------------
    if args.skip_training:
        print(f"\nStep 2: Skipping GAN training — loading {args.gan_checkpoint}")
        if not Path(args.gan_checkpoint).exists():
            print(f"  ERROR: {args.gan_checkpoint} not found.")
            print("  Run without --skip-training to train the GAN first.")
            exit(1)
    else:
        print(f"\nStep 2: Training Conditional GAN on real CT data ({args.epochs} epochs)...")

        dataset = GANTrainDataset(args.data_dir, img_size=GAN_IMG_SIZE)
        loader  = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=0, pin_memory=True)

        print(f"  Total real training images: {len(dataset)}")
        print(f"  Batches per epoch         : {len(loader)}")
        print(f"  GAN image size            : {GAN_IMG_SIZE}x{GAN_IMG_SIZE} "
              f"(upscaled to {IMAGE_SIZE}x{IMAGE_SIZE} when saved)")

        gan_config = {
            "latent_dim"          : GAN_LATENT_DIM,
            "num_classes"         : NUM_CLASSES,
            "img_size"            : GAN_IMG_SIZE,
            "lr"                  : GAN_LR,
            "beta1"               : 0.5,
            "beta2"               : 0.999,
            "checkpoint_dir"      : "checkpoints/gan",
            "sample_dir"          : "outputs/gan/samples",
            "output_dir"          : "outputs/gan",
            "checkpoint_interval" : 25,
            "sample_interval"     : 25,
        }

        ensure_dir(gan_config["checkpoint_dir"])
        ensure_dir(gan_config["sample_dir"])
        ensure_dir(gan_config["output_dir"])

        trainer = GANTrainer(gan_config)
        trainer.train(loader, num_epochs=args.epochs)

        args.gan_checkpoint = "checkpoints/gan/gan_best.pth"

    # ------------------------------------------------------------------
    # Step 3: Generate synthetic images
    # ------------------------------------------------------------------
    print(f"\nStep 3: Generating {n_to_generate} synthetic '{minority_class}' images...")
    generator, gan_config = load_generator(args.gan_checkpoint, device)
    raw_images = generate_images(
        generator, gan_config,
        num_samples=n_to_generate,
        target_class=minority_idx,
        device=device
    )
    print(f"  Generated {len(raw_images)} images at {GAN_IMG_SIZE}x{GAN_IMG_SIZE}")

    # ------------------------------------------------------------------
    # Step 4: Resize to IMAGE_SIZE and save directly into training folder
    # ------------------------------------------------------------------
    output_dir = Path(args.data_dir) / minority_class
    print(f"\nStep 4: Resizing to {IMAGE_SIZE}x{IMAGE_SIZE} and saving to {output_dir}...")
    saved = save_synthetic_images(raw_images, output_dir, target_size=IMAGE_SIZE)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    new_counts = count_real_images(args.data_dir)
    total_cancer    = cancer_n    + (saved if minority_class == "cancer"    else 0)
    total_no_cancer = no_cancer_n + (saved if minority_class == "no_cancer" else 0)

    print("\n" + "=" * 70)
    print("AUGMENTATION COMPLETE")
    print("=" * 70)
    print(f"  Synthetic images saved  : {saved}")
    print(f"  New class distribution  :")
    print(f"    no_cancer (real only) : {no_cancer_n}")
    print(f"    cancer    (real only) : {cancer_n}")
    print(f"    no_cancer (with synth): {total_no_cancer}")
    print(f"    cancer    (with synth): {total_cancer}")
    print()
    print("Next steps:")
    print("  1. Train base models on augmented data:")
    print("       python train.py --model efficientnet --epochs 50 --data-dir data/kaggle_lung_cancer")
    print("       python train.py --model densenet     --epochs 50 --data-dir data/kaggle_lung_cancer")
    print("       python train.py --model resnet       --epochs 50 --data-dir data/kaggle_lung_cancer")
    print("  2. Train the stacked ensemble (meta-learner on validation set):")
    print("       python train_ensemble.py")
    print("  3. Generate Grad-CAM visualisations:")
    print("       python generate_gradcam.py")


if __name__ == "__main__":
    main()
