"""
Train the stacked ensemble using all 3 base models.
Generates base model predictions and trains XGBoost meta-learner.
"""
import torch
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm

from configs.config import *
from src.utils import set_seed, ensure_dir, load_checkpoint
from src.datasets import LungCancerDataset
from src.augmentations import get_val_transforms
from src.models import get_model
from src.ensemble.meta_models import MetaLearner
from src.evaluation import ModelEvaluator
from torch.utils.data import DataLoader

set_seed(42)

DATA_DIR = "data/kaggle_lung_cancer/train"
TEST_DIR = "data/kaggle_lung_cancer/test"

print("=" * 70)
print("STACKED ENSEMBLE TRAINING")
print("=" * 70)

# ---- Step 1: Load all training image paths and labels ----
print("\nStep 1: Loading training data...")
train_dir = Path(DATA_DIR)
train_images = []
train_labels = []

for class_idx, class_name in enumerate(["no_cancer", "cancer"]):
    class_dir = train_dir / class_name
    if class_dir.exists():
        for img_path in sorted(class_dir.glob("*")):
            if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                train_images.append(str(img_path))
                train_labels.append(class_idx)

train_images = np.array(train_images)
train_labels = np.array(train_labels)
print(f"  Total training images: {len(train_images)}")
print(f"  No Cancer: {(train_labels == 0).sum()}, Cancer: {(train_labels == 1).sum()}")

# ---- Step 2: Generate base model predictions on training data ----
print("\nStep 2: Generating base model predictions...")

model_names = ["efficientnet", "densenet", "resnet"]
checkpoint_paths = {
    "efficientnet": "checkpoints/efficientnet_finetuned_best.pth",
    "densenet": "checkpoints/densenet_finetuned_best.pth",
    "resnet": "checkpoints/resnet_finetuned_best.pth",
}

# Check all checkpoints exist
missing = False
for name, path in checkpoint_paths.items():
    if not Path(path).exists():
        print(f"  ❌ ERROR: {path} not found! Train {name} first.")
        missing = True
    else:
        print(f"  ✓ Found: {path}")

if missing:
    print("\n❌ Cannot train ensemble — missing base model checkpoints.")
    print("   Run the following commands first:")
    print("   python train.py --model densenet --epochs 50 --data-dir data/kaggle_lung_cancer")
    print("   python train.py --model resnet --epochs 50 --data-dir data/kaggle_lung_cancer")
    exit(1)

device = DEVICE
val_transform = get_val_transforms(IMAGE_SIZE)

# Get predictions from each model on training data
all_train_probs = []

for model_name in model_names:
    print(f"\n  Getting predictions from {model_name}...")
    model = get_model(model_name, num_classes=NUM_CLASSES, pretrained=False)
    load_checkpoint(model, None, checkpoint_paths[model_name])
    model = model.to(device)
    model.eval()

    dataset = LungCancerDataset(
        image_paths=train_images.tolist(),
        labels=train_labels.tolist(),
        transform=val_transform
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model_probs = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"  {model_name}"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            model_probs.append(probs)

    model_probs = np.concatenate(model_probs, axis=0)
    all_train_probs.append(model_probs)
    print(f"  {model_name}: {model_probs.shape}")

# Stack all predictions as meta-features
# Shape: (num_samples, num_models * num_classes) = (767, 6)
meta_features_train = np.hstack(all_train_probs)
print(f"\nMeta-features shape: {meta_features_train.shape}")

# ---- Step 3: Train meta-learner ----
print("\nStep 3: Training XGBoost meta-learner...")

meta_learner = MetaLearner("xgboost")
meta_learner.fit(meta_features_train, train_labels)

# Save meta-learner
ensure_dir("checkpoints/ensemble")
joblib.dump(meta_learner, "checkpoints/ensemble/meta_learner_xgboost.pkl")
print("  ✓ Meta-learner saved to: checkpoints/ensemble/meta_learner_xgboost.pkl")

# ---- Step 4: Evaluate ensemble on test set ----
print("\nStep 4: Evaluating ensemble on test set...")

test_dir_path = Path(TEST_DIR)
test_images_list = []
test_labels_list = []

for class_idx, class_name in enumerate(["no_cancer", "cancer"]):
    class_dir = test_dir_path / class_name
    if class_dir.exists():
        for img_path in sorted(class_dir.glob("*")):
            if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                test_images_list.append(str(img_path))
                test_labels_list.append(class_idx)

test_labels_arr = np.array(test_labels_list)
print(f"  Test set: {len(test_images_list)} images")

# Get base model predictions on test set
all_test_probs = []

for model_name in model_names:
    print(f"  Getting test predictions from {model_name}...")
    model = get_model(model_name, num_classes=NUM_CLASSES, pretrained=False)
    load_checkpoint(model, None, checkpoint_paths[model_name])
    model = model.to(device)
    model.eval()

    dataset = LungCancerDataset(
        image_paths=test_images_list,
        labels=test_labels_list,
        transform=val_transform
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model_probs = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"  {model_name}"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            model_probs.append(probs)

    model_probs = np.concatenate(model_probs, axis=0)
    all_test_probs.append(model_probs)

meta_features_test = np.hstack(all_test_probs)

# Ensemble prediction
ensemble_preds = meta_learner.predict(meta_features_test)
ensemble_proba = meta_learner.predict_proba(meta_features_test)

# Also get individual model predictions for comparison
individual_results = {}
for i, model_name in enumerate(model_names):
    preds = np.argmax(all_test_probs[i], axis=1)
    evaluator = ModelEvaluator(test_labels_arr, preds, all_test_probs[i])
    individual_results[model_name] = evaluator.calculate_metrics()

# Ensemble evaluation
print(f"\n{'=' * 70}")
print("ENSEMBLE EVALUATION ON TEST SET")
print(f"{'=' * 70}")

ensemble_evaluator = ModelEvaluator(test_labels_arr, ensemble_preds, ensemble_proba)
ensemble_metrics = ensemble_evaluator.print_report()
ensure_dir("outputs/test_evaluation/ensemble")
ensemble_evaluator.save_all_plots("outputs/test_evaluation/ensemble")

# ---- Step 5: Print comparison ----
print("\n" + "=" * 70)
print("FINAL COMPARISON — TEST SET")
print("=" * 70)
print(f"\n{'Model':<18} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
print("-" * 78)

for model_name, metrics in individual_results.items():
    print(f"{model_name:<18} "
          f"{metrics['accuracy']*100:<12.2f} "
          f"{metrics['precision']:<12.4f} "
          f"{metrics['recall']:<12.4f} "
          f"{metrics['f1']:<12.4f} "
          f"{metrics.get('roc_auc', 0):<12.4f}")

print(f"{'ENSEMBLE (XGB)':<18} "
      f"{ensemble_metrics['accuracy']*100:<12.2f} "
      f"{ensemble_metrics['precision']:<12.4f} "
      f"{ensemble_metrics['recall']:<12.4f} "
      f"{ensemble_metrics['f1']:<12.4f} "
      f"{ensemble_metrics.get('roc_auc', 0):<12.4f}")

# Improvement
best_individual_acc = max(m['accuracy'] for m in individual_results.values())
improvement = (ensemble_metrics['accuracy'] - best_individual_acc) * 100
print(f"\n{'Ensemble improvement over best model:':<40} {improvement:+.2f}%")

print("\n✓ Ensemble training and evaluation complete!")
print("✓ Meta-learner saved to: checkpoints/ensemble/meta_learner_xgboost.pkl")
print("✓ Results saved to: outputs/test_evaluation/ensemble/")
