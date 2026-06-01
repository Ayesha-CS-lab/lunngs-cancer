"""
Evaluate all trained models on the held-out test set.
Run this ONLY ONCE after all training is complete.
"""
import torch
import numpy as np
from pathlib import Path
from configs.config import *
from src.utils import set_seed
from src.datasets import LungCancerDataset
from src.augmentations import get_val_transforms
from src.models import get_model
from src.evaluation import evaluate_model, ModelEvaluator
from src.utils import load_checkpoint
from torch.utils.data import DataLoader

set_seed(42)

# Test data directory
TEST_DATA_DIR = "data/kaggle_lung_cancer"

# Models to evaluate
models_to_evaluate = {
    "efficientnet": "checkpoints/efficientnet_finetuned_best.pth",
    "densenet": "checkpoints/densenet_finetuned_best.pth",
    "resnet": "checkpoints/resnet_finetuned_best.pth",
}

print("=" * 70)
print("TEST SET EVALUATION — ALL MODELS")
print("=" * 70)

# Create test data loader
test_dir = Path(TEST_DATA_DIR) / "test"
test_images = []
test_labels = []

for class_idx, class_name in enumerate(["no_cancer", "cancer"]):
    class_dir = test_dir / class_name
    if class_dir.exists():
        for img_path in sorted(class_dir.glob("*")):
            if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                test_images.append(str(img_path))
                test_labels.append(class_idx)

print(f"\nTest set: {len(test_images)} images")
print(f"  No Cancer: {test_labels.count(0)}")
print(f"  Cancer: {test_labels.count(1)}")

test_dataset = LungCancerDataset(
    image_paths=test_images,
    labels=test_labels,
    transform=get_val_transforms(IMAGE_SIZE)
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Evaluate each model
all_results = {}

for model_name, checkpoint_path in models_to_evaluate.items():
    if not Path(checkpoint_path).exists():
        print(f"\n⚠️  Skipping {model_name} — checkpoint not found: {checkpoint_path}")
        continue

    print(f"\n{'=' * 70}")
    print(f"Evaluating {model_name.upper()} on TEST SET")
    print(f"{'=' * 70}")

    model = get_model(model_name, num_classes=NUM_CLASSES, pretrained=False)
    load_checkpoint(model, None, checkpoint_path)

    metrics = evaluate_model(
        model,
        test_loader,
        device=DEVICE,
        output_dir=f"outputs/test_evaluation/{model_name}"
    )

    all_results[model_name] = metrics

# Summary
print("\n" + "=" * 70)
print("SUMMARY — TEST SET RESULTS")
print("=" * 70)
print(f"\n{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
print("-" * 75)

for model_name, metrics in all_results.items():
    print(f"{model_name:<15} "
          f"{metrics['accuracy']*100:<12.2f} "
          f"{metrics['precision']:<12.4f} "
          f"{metrics['recall']:<12.4f} "
          f"{metrics['f1']:<12.4f} "
          f"{metrics.get('roc_auc', 0):<12.4f}")

print("\n✓ Test evaluation complete!")
print(f"✓ Plots saved to: outputs/test_evaluation/")
