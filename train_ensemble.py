"""
Train the stacked ensemble using all 3 base models.

FIX: Meta-learner is trained on VALIDATION set predictions (not training set)
to prevent data leakage. Base models have memorized the training set, so
using training predictions gave artificially perfect 100% results.

Supports both datasets via command-line arguments:
    python train_ensemble.py                          # IQ-OTH/NCCD (default)
    python train_ensemble.py \\
        --val-dir  data/lidc_idri/val  \\
        --test-dir data/lidc_idri/test \\
        --checkpoint-dir checkpoints/lidc_idri \\
        --output-dir outputs/lidc_idri
"""
import argparse
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

parser = argparse.ArgumentParser(description="Train stacked ensemble")
parser.add_argument("--train-dir",      type=str, default="data/kaggle_lung_cancer/train")
parser.add_argument("--val-dir",        type=str, default="data/kaggle_lung_cancer/val")
parser.add_argument("--test-dir",       type=str, default="data/kaggle_lung_cancer/test")
parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                    help="Directory that contains efficientnet/densenet/resnet checkpoints")
parser.add_argument("--output-dir",     type=str, default="outputs/test_evaluation",
                    help="Directory for evaluation plots")
parser.add_argument("--seed",           type=int, default=42)
args = parser.parse_args()

set_seed(args.seed)

TRAIN_DIR = args.train_dir
VAL_DIR   = args.val_dir
TEST_DIR  = args.test_dir
CKPT_DIR  = args.checkpoint_dir
OUT_DIR   = args.output_dir

print("=" * 70)
print("STACKED ENSEMBLE TRAINING")
print("=" * 70)

# ---- Step 1: Show dataset statistics ----
print("\nStep 1: Dataset overview...")
for split_name, split_dir in [("Train", TRAIN_DIR), ("Val", VAL_DIR), ("Test", TEST_DIR)]:
    no_c = len(list((Path(split_dir) / "no_cancer").glob("*"))) if (Path(split_dir) / "no_cancer").exists() else 0
    can  = len(list((Path(split_dir) / "cancer").glob("*")))    if (Path(split_dir) / "cancer").exists() else 0
    print(f"  {split_name}: no_cancer={no_c}, cancer={can}, total={no_c + can}")

# ---- Step 2: Load validation images (meta-learner training data) ----
print("\nStep 2: Loading validation set for meta-learner training...")
val_dir = Path(VAL_DIR)
val_images = []
val_labels_meta = []

for class_idx, class_name in enumerate(["no_cancer", "cancer"]):
    class_dir = val_dir / class_name
    if class_dir.exists():
        for img_path in sorted(class_dir.glob("*")):
            if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                val_images.append(str(img_path))
                val_labels_meta.append(class_idx)

val_labels_meta = np.array(val_labels_meta)
print(f"  Validation images: {len(val_images)}")
print(f"  No Cancer: {(val_labels_meta == 0).sum()}, Cancer: {(val_labels_meta == 1).sum()}")

# ---- Step 3: Check base model checkpoints ----
model_names = ["efficientnet", "densenet", "resnet"]
checkpoint_paths = {
    "efficientnet": f"{CKPT_DIR}/efficientnet_finetuned_best.pth",
    "densenet":     f"{CKPT_DIR}/densenet_finetuned_best.pth",
    "resnet":       f"{CKPT_DIR}/resnet_finetuned_best.pth",
}

print("\nStep 3: Checking base model checkpoints...")
missing = False
for name, path in checkpoint_paths.items():
    if not Path(path).exists():
        print(f"  ERROR: {path} not found! Train {name} first.")
        missing = True
    else:
        print(f"  Found: {path}")

if missing:
    print("\nCannot train ensemble — missing base model checkpoints.")
    print("Run these commands first:")
    for name in model_names:
        print(f"  python train.py --model {name} --epochs 50 --data-dir data/kaggle_lung_cancer")
    exit(1)

device = DEVICE
val_transform = get_val_transforms(IMAGE_SIZE)

# ---- Step 4: Get base model predictions on VALIDATION set ----
#
# This is the key fix: we predict on the validation set (data the base models
# have NOT been trained on) so the meta-learner learns from genuinely
# unseen signal, not memorized training patterns.
#
print("\nStep 4: Generating base model predictions on validation set...")
all_val_probs = []

for model_name in model_names:
    print(f"\n  Getting predictions from {model_name}...")
    model = get_model(model_name, num_classes=NUM_CLASSES, pretrained=False)
    load_checkpoint(model, None, checkpoint_paths[model_name])
    model = model.to(device)
    model.eval()

    dataset = LungCancerDataset(
        image_paths=val_images,
        labels=val_labels_meta.tolist(),
        transform=val_transform
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model_probs = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc=f"  {model_name}"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            model_probs.append(probs)

    model_probs = np.concatenate(model_probs, axis=0)
    all_val_probs.append(model_probs)
    val_acc = np.mean(np.argmax(model_probs, axis=1) == val_labels_meta) * 100
    print(f"  {model_name} validation accuracy: {val_acc:.2f}%")

# Stack predictions: shape (num_val_samples, num_models * num_classes) = (164, 6)
meta_features_val = np.hstack(all_val_probs)
print(f"\nMeta-features shape: {meta_features_val.shape}")

# ---- Step 5: Train meta-learner on validation predictions ----
print("\nStep 5: Training XGBoost meta-learner on validation predictions...")
meta_learner = MetaLearner("xgboost")
meta_learner.fit(meta_features_val, val_labels_meta)

ensemble_ckpt_dir = f"{CKPT_DIR}/ensemble"
ensure_dir(ensemble_ckpt_dir)
meta_path = f"{ensemble_ckpt_dir}/meta_learner_xgboost.pkl"
joblib.dump(meta_learner, meta_path)
print(f"  Meta-learner saved to: {meta_path}")

# ---- Step 6: Evaluate ensemble on held-out test set ----
print("\nStep 6: Evaluating ensemble on test set...")

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

all_test_probs = []
individual_results = {}

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
        for images, _ in tqdm(loader, desc=f"  {model_name}"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            model_probs.append(probs)

    model_probs = np.concatenate(model_probs, axis=0)
    all_test_probs.append(model_probs)

    preds = np.argmax(model_probs, axis=1)
    evaluator = ModelEvaluator(test_labels_arr, preds, model_probs)
    individual_results[model_name] = evaluator.calculate_metrics()

meta_features_test = np.hstack(all_test_probs)
ensemble_preds = meta_learner.predict(meta_features_test)
ensemble_proba = meta_learner.predict_proba(meta_features_test)

print(f"\n{'=' * 70}")
print("ENSEMBLE EVALUATION ON TEST SET")
print(f"{'=' * 70}")

ensemble_evaluator = ModelEvaluator(test_labels_arr, ensemble_preds, ensemble_proba)
ensemble_metrics = ensemble_evaluator.print_report()
ensemble_out = f"{OUT_DIR}/ensemble"
ensure_dir(ensemble_out)
ensemble_evaluator.save_all_plots(ensemble_out)

# ---- Step 7: Final comparison table ----
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

best_individual_acc = max(m['accuracy'] for m in individual_results.values())
improvement = (ensemble_metrics['accuracy'] - best_individual_acc) * 100
print(f"\n{'Ensemble improvement over best individual model:':<48} {improvement:+.2f}%")

print("\nEnsemble training and evaluation complete!")
print(f"Meta-learner saved to: {meta_path}")
print(f"Results saved to: {ensemble_out}/")
