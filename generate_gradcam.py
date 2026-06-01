"""
Generate Grad-CAM heatmaps on real test images.
Creates visual explanations for model predictions.
"""
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from configs.config import *
from src.utils import set_seed, ensure_dir, load_checkpoint
from src.models import get_model
from src.augmentations import get_val_transforms
from src.explainability.gradcam import visualize_gradcam
from src.preprocessing import load_image

set_seed(42)

DEVICE_STR = 'cuda' if torch.cuda.is_available() else 'cpu'
TEST_DIR = Path("data/kaggle_lung_cancer/test")
OUTPUT_DIR = Path("outputs/heatmaps")
ensure_dir(str(OUTPUT_DIR))

NUM_SAMPLES = 5  # Number of images per class to visualize

print("=" * 70)
print("GRAD-CAM HEATMAP GENERATION")
print("=" * 70)

# Collect sample images
cancer_images = sorted(list((TEST_DIR / "cancer").glob("*")))[:NUM_SAMPLES]
no_cancer_images = sorted(list((TEST_DIR / "no_cancer").glob("*")))[:NUM_SAMPLES]

print(f"\nSelected {len(cancer_images)} cancer + {len(no_cancer_images)} no_cancer images")

# Load EfficientNet model (best trained model)
model_name = "efficientnet"
checkpoint_path = "checkpoints/efficientnet_finetuned_best.pth"

if not Path(checkpoint_path).exists():
    print(f"❌ Checkpoint not found: {checkpoint_path}")
    print("   Train EfficientNet first: python train.py --model efficientnet --epochs 50 --data-dir data/kaggle_lung_cancer")
    exit(1)

print(f"\nLoading {model_name} from {checkpoint_path}...")
model = get_model(model_name, num_classes=NUM_CLASSES, pretrained=False)
load_checkpoint(model, None, checkpoint_path)
model = model.to(DEVICE_STR)
model.eval()

transform = get_val_transforms(IMAGE_SIZE)

# Generate heatmaps for cancer images
print("\n--- Cancer Images ---")
for i, img_path in enumerate(cancer_images):
    print(f"  Processing: {img_path.name}")

    image = load_image(str(img_path), image_size=IMAGE_SIZE)
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(DEVICE_STR)

    save_path = str(OUTPUT_DIR / f"cancer_{i+1}_{img_path.stem}_gradcam.png")

    overlaid, heatmap, pred_class = visualize_gradcam(
        model=model,
        model_name=model_name,
        image=image,
        image_tensor=image_tensor,
        save_path=save_path
    )

    class_name = "Cancer" if pred_class == 1 else "No Cancer"
    print(f"    Predicted: {class_name} | Saved: {save_path}")

# Generate heatmaps for no-cancer images
print("\n--- No Cancer Images ---")
for i, img_path in enumerate(no_cancer_images):
    print(f"  Processing: {img_path.name}")

    image = load_image(str(img_path), image_size=IMAGE_SIZE)
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(DEVICE_STR)

    save_path = str(OUTPUT_DIR / f"no_cancer_{i+1}_{img_path.stem}_gradcam.png")

    overlaid, heatmap, pred_class = visualize_gradcam(
        model=model,
        model_name=model_name,
        image=image,
        image_tensor=image_tensor,
        save_path=save_path
    )

    class_name = "Cancer" if pred_class == 1 else "No Cancer"
    print(f"    Predicted: {class_name} | Saved: {save_path}")

# ---- Create a comparison grid ----
print("\n--- Creating Comparison Grid ---")

fig, axes = plt.subplots(2, NUM_SAMPLES, figsize=(4 * NUM_SAMPLES, 8))
fig.suptitle("Grad-CAM Heatmap Comparison", fontsize=16, fontweight='bold')

for i, img_path in enumerate(cancer_images):
    image = load_image(str(img_path), image_size=IMAGE_SIZE)
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(DEVICE_STR)

    overlaid, _, pred_class = visualize_gradcam(
        model=model,
        model_name=model_name,
        image=image,
        image_tensor=image_tensor
    )

    axes[0, i].imshow(overlaid)
    axes[0, i].set_title(f"Cancer #{i+1}\nPred: {'Cancer' if pred_class == 1 else 'No Cancer'}", fontsize=9)
    axes[0, i].axis('off')

for i, img_path in enumerate(no_cancer_images):
    image = load_image(str(img_path), image_size=IMAGE_SIZE)
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(DEVICE_STR)

    overlaid, _, pred_class = visualize_gradcam(
        model=model,
        model_name=model_name,
        image=image,
        image_tensor=image_tensor
    )

    axes[1, i].imshow(overlaid)
    axes[1, i].set_title(f"No Cancer #{i+1}\nPred: {'Cancer' if pred_class == 1 else 'No Cancer'}", fontsize=9)
    axes[1, i].axis('off')

axes[0, 0].set_ylabel("Cancer\n(Ground Truth)", fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel("No Cancer\n(Ground Truth)", fontsize=12, fontweight='bold')

plt.tight_layout()
grid_path = str(OUTPUT_DIR / "gradcam_comparison_grid.png")
plt.savefig(grid_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Comparison grid saved to: {grid_path}")

# ---- Multi-model comparison (if all models available) ----
model_configs = {
    "efficientnet": "checkpoints/efficientnet_finetuned_best.pth",
    "densenet": "checkpoints/densenet_finetuned_best.pth",
    "resnet": "checkpoints/resnet_finetuned_best.pth",
}

available_models = {k: v for k, v in model_configs.items() if Path(v).exists()}

if len(available_models) >= 2:
    print("\n--- Multi-Model Grad-CAM Comparison ---")

    # Use first cancer image
    sample_img_path = cancer_images[0]
    sample_image = load_image(str(sample_img_path), image_size=IMAGE_SIZE)
    transformed = transform(image=sample_image)
    sample_tensor = transformed['image'].unsqueeze(0).to(DEVICE_STR)

    fig, axes = plt.subplots(1, len(available_models) + 1, figsize=(5 * (len(available_models) + 1), 5))
    fig.suptitle(f"Grad-CAM Comparison Across Models\n({sample_img_path.name})", fontsize=14, fontweight='bold')

    # Show original
    display_img = sample_image if sample_image.max() > 1 else (sample_image * 255).astype(np.uint8)
    axes[0].imshow(display_img)
    axes[0].set_title("Original")
    axes[0].axis('off')

    for idx, (mname, mpath) in enumerate(available_models.items()):
        m = get_model(mname, num_classes=NUM_CLASSES, pretrained=False)
        load_checkpoint(m, None, mpath)
        m = m.to(DEVICE_STR)
        m.eval()

        overlaid, _, pred_class = visualize_gradcam(
            model=m,
            model_name=mname,
            image=sample_image,
            image_tensor=sample_tensor
        )

        axes[idx + 1].imshow(overlaid)
        axes[idx + 1].set_title(f"{mname}\nPred: {'Cancer' if pred_class == 1 else 'No Cancer'}")
        axes[idx + 1].axis('off')

    plt.tight_layout()
    multi_path = str(OUTPUT_DIR / "multi_model_gradcam_comparison.png")
    plt.savefig(multi_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Multi-model comparison saved to: {multi_path}")
else:
    print(f"\n--- Skipping multi-model comparison (only {len(available_models)} model(s) available) ---")

print("\n✓ Grad-CAM generation complete!")
print(f"✓ All heatmaps saved to: {OUTPUT_DIR}/")
