"""
Dataset Comparison Script
=========================
Evaluates trained models on both IQ-OTH/NCCD and LIDC-IDRI test sets,
then generates a side-by-side comparison report with plots.

Run AFTER training models on both datasets:
    python compare_datasets.py

Outputs:
    outputs/comparison/comparison_table.csv
    outputs/comparison/accuracy_comparison.png
    outputs/comparison/auc_comparison.png
    outputs/comparison/full_metrics_heatmap.png
    outputs/comparison/cross_dataset_summary.txt
"""

import sys
import numpy as np
import torch
import joblib
import csv
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from configs.config import *
from src.utils import set_seed, load_checkpoint
from src.datasets import LungCancerDataset
from src.augmentations import get_val_transforms
from src.models import get_model
from src.evaluation import ModelEvaluator

set_seed(42)

# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------
DATASETS = {
    "IQ-OTH/NCCD": {
        "test_dir":      "data/kaggle_lung_cancer/test",
        "val_dir":       "data/kaggle_lung_cancer/val",
        "checkpoint_dir": "checkpoints",
        "color":         "#2196F3",
    },
    "LIDC-IDRI": {
        "test_dir":      "data/lidc_idri/test",
        "val_dir":       "data/lidc_idri/val",
        "checkpoint_dir": "checkpoints/lidc_idri",
        "color":         "#FF5722",
    },
}

MODELS    = ["efficientnet", "densenet", "resnet"]
MODEL_DISPLAY = {
    "efficientnet": "EfficientNet-B0",
    "densenet":     "DenseNet-121",
    "resnet":       "ResNet-50",
    "ensemble":     "Ensemble (XGBoost)",
}
OUT_DIR   = "outputs/comparison"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_images_and_labels(test_dir):
    test_dir = Path(test_dir)
    images, labels = [], []
    for class_idx, class_name in enumerate(["no_cancer", "cancer"]):
        class_dir = test_dir / class_name
        if class_dir.exists():
            for p in sorted(class_dir.glob("*")):
                if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    images.append(str(p))
                    labels.append(class_idx)
    return images, np.array(labels)


def get_model_predictions(model_name, ckpt_dir, test_images, test_labels):
    """Load a trained model and return predictions on the test set."""
    ckpt_path = Path(ckpt_dir) / f"{model_name}_finetuned_best.pth"
    if not ckpt_path.exists():
        print(f"    Checkpoint not found: {ckpt_path}")
        return None, None

    model = get_model(model_name, num_classes=NUM_CLASSES, pretrained=False)
    load_checkpoint(model, None, str(ckpt_path))
    model = model.to(DEVICE)
    model.eval()

    dataset = LungCancerDataset(
        image_paths=test_images,
        labels=test_labels.tolist(),
        transform=get_val_transforms(IMAGE_SIZE),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_probs = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)
    preds = np.argmax(all_probs, axis=1)
    return preds, all_probs


def get_ensemble_predictions(ckpt_dir, test_images, test_labels):
    """Load XGBoost meta-learner and all base models, return ensemble preds."""
    meta_path = Path(ckpt_dir) / "ensemble" / "meta_learner_xgboost.pkl"
    if not meta_path.exists():
        print(f"    Ensemble meta-learner not found: {meta_path}")
        return None, None

    meta_learner = joblib.load(str(meta_path))

    all_base_probs = []
    for model_name in MODELS:
        _, probs = get_model_predictions(model_name, ckpt_dir, test_images, test_labels)
        if probs is None:
            print(f"    Skipping ensemble — missing {model_name} checkpoint.")
            return None, None
        all_base_probs.append(probs)

    meta_features = np.hstack(all_base_probs)
    preds = meta_learner.predict(meta_features)
    proba = meta_learner.predict_proba(meta_features)
    return preds, proba


def compute_metrics(true_labels, preds, proba):
    evaluator = ModelEvaluator(true_labels, preds, proba)
    m = evaluator.calculate_metrics()
    return {
        "accuracy":   round(m.get("accuracy",  0) * 100, 2),
        "precision":  round(m.get("precision", 0) * 100, 2),
        "recall":     round(m.get("recall",    0) * 100, 2),
        "f1":         round(m.get("f1",        0) * 100, 2),
        "roc_auc":    round(m.get("roc_auc",   0) * 100, 2),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_accuracy_comparison(results, save_path):
    """Grouped bar chart: accuracy per model, grouped by dataset."""
    fig, ax = plt.subplots(figsize=(12, 6))

    model_keys   = MODELS + ["ensemble"]
    model_labels = [MODEL_DISPLAY[m] for m in model_keys]
    n_models     = len(model_keys)
    n_datasets   = len(DATASETS)
    width        = 0.35
    x            = np.arange(n_models)

    for i, (ds_name, ds_cfg) in enumerate(DATASETS.items()):
        accuracies = []
        for mk in model_keys:
            acc = results.get(ds_name, {}).get(mk, {}).get("accuracy", 0)
            accuracies.append(acc)
        offset = (i - n_datasets / 2 + 0.5) * width
        bars   = ax.bar(x + offset, accuracies, width, label=ds_name,
                        color=ds_cfg["color"], alpha=0.85, edgecolor="black")
        for bar, val in zip(bars, accuracies):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{val:.1f}%", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Model Accuracy Comparison: IQ-OTH/NCCD vs LIDC-IDRI",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=97, color="green", linestyle="--", alpha=0.5,
               label="Target (97%)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_metrics_heatmap(results, save_path):
    """Heatmap of all metrics for all models and datasets."""
    metrics_order = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC AUC"]
    model_keys    = MODELS + ["ensemble"]

    rows = []
    row_labels = []
    for ds_name in DATASETS:
        for mk in model_keys:
            row = [results.get(ds_name, {}).get(mk, {}).get(met, 0)
                   for met in metrics_order]
            rows.append(row)
            row_labels.append(f"{ds_name}\n{MODEL_DISPLAY[mk]}")

    data = np.array(rows, dtype=float)

    fig, ax = plt.subplots(figsize=(10, max(6, len(row_labels) * 0.55 + 1)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=50, vmax=100)

    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    for i in range(len(row_labels)):
        for j in range(len(metric_labels)):
            val = data[i, j]
            color = "white" if val < 65 or val > 90 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, label="Score (%)", shrink=0.8)
    ax.set_title("Full Metrics Heatmap — Both Datasets",
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_auc_comparison(results, save_path):
    """ROC AUC bar chart side by side."""
    fig, ax = plt.subplots(figsize=(12, 6))

    model_keys   = MODELS + ["ensemble"]
    model_labels = [MODEL_DISPLAY[m] for m in model_keys]
    x            = np.arange(len(model_keys))
    width        = 0.35

    for i, (ds_name, ds_cfg) in enumerate(DATASETS.items()):
        aucs   = [results.get(ds_name, {}).get(mk, {}).get("roc_auc", 0)
                  for mk in model_keys]
        offset = (i - 1) * width / 2 + width / 4
        bars   = ax.bar(x + offset * 2, aucs, width,
                        label=ds_name, color=ds_cfg["color"],
                        alpha=0.85, edgecolor="black")
        for bar, val in zip(bars, aucs):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3,
                        f"{val:.1f}%", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("ROC AUC (%)", fontsize=12)
    ax.set_title("ROC AUC Comparison: IQ-OTH/NCCD vs LIDC-IDRI",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def save_csv(results, save_path):
    rows = []
    for ds_name in DATASETS:
        for mk in MODELS + ["ensemble"]:
            m = results.get(ds_name, {}).get(mk, {})
            rows.append({
                "dataset":   ds_name,
                "model":     MODEL_DISPLAY.get(mk, mk),
                "accuracy":  m.get("accuracy",  "-"),
                "precision": m.get("precision", "-"),
                "recall":    m.get("recall",    "-"),
                "f1":        m.get("f1",        "-"),
                "roc_auc":   m.get("roc_auc",   "-"),
            })
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {save_path}")


def print_comparison_table(results):
    header = f"\n{'Model':<22} {'IQ-OTH Acc':>12} {'LIDC Acc':>10} {'IQ-OTH AUC':>12} {'LIDC AUC':>10} {'Harder?':>10}"
    print(header)
    print("-" * len(header))

    for mk in MODELS + ["ensemble"]:
        iq_m  = results.get("IQ-OTH/NCCD", {}).get(mk, {})
        li_m  = results.get("LIDC-IDRI",   {}).get(mk, {})
        iq_ac = iq_m.get("accuracy",  0)
        li_ac = li_m.get("accuracy",  0)
        iq_au = iq_m.get("roc_auc",   0)
        li_au = li_m.get("roc_auc",   0)
        harder = "LIDC-IDRI" if (iq_ac > li_ac or iq_au > li_au) else "IQ-OTH/NCCD"

        iq_ac_s = f"{iq_ac:.1f}%" if iq_ac else "-"
        li_ac_s = f"{li_ac:.1f}%" if li_ac else "-"
        iq_au_s = f"{iq_au:.1f}%" if iq_au else "-"
        li_au_s = f"{li_au:.1f}%" if li_au else "-"

        print(f"{MODEL_DISPLAY.get(mk, mk):<22} {iq_ac_s:>12} {li_ac_s:>10} "
              f"{iq_au_s:>12} {li_au_s:>10} {harder:>10}")


def save_text_summary(results, save_path):
    lines = ["CROSS-DATASET COMPARISON SUMMARY", "=" * 60, ""]
    lines.append("Datasets:")
    lines.append("  IQ-OTH/NCCD — Iraqi-Oncology-Teaching-Hospital / NCI dataset")
    lines.append("              — Clean, well-structured CT scan images")
    lines.append("  LIDC-IDRI   — Lung Image Database Consortium")
    lines.append("              — Larger, more diverse, community-annotated")
    lines.append("")
    lines.append("Models compared: EfficientNet-B0, DenseNet-121, ResNet-50, Ensemble")
    lines.append("")
    lines.append("Results (Accuracy / ROC AUC):")

    for mk in MODELS + ["ensemble"]:
        iq_m  = results.get("IQ-OTH/NCCD", {}).get(mk, {})
        li_m  = results.get("LIDC-IDRI",   {}).get(mk, {})
        lines.append(
            f"  {MODEL_DISPLAY.get(mk, mk):<22} "
            f"IQ-OTH: {iq_m.get('accuracy', 0):.1f}% / {iq_m.get('roc_auc', 0):.1f}%  |  "
            f"LIDC: {li_m.get('accuracy', 0):.1f}% / {li_m.get('roc_auc', 0):.1f}%"
        )

    lines.append("")
    lines.append("Interpretation:")
    lines.append("  - If LIDC-IDRI accuracy is lower, it confirms better generalisation")
    lines.append("    challenge — the model is not over-fitting to one dataset.")
    lines.append("  - The Ensemble should outperform individual models on both datasets.")
    lines.append("  - If IQ-OTH scores are still near-perfect, consider increasing")
    lines.append("    regularisation or using cross-dataset validation.")

    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CROSS-DATASET COMPARISON")
    print("=" * 70)

    results = {}
    available_datasets = []

    for ds_name, ds_cfg in DATASETS.items():
        test_dir  = ds_cfg["test_dir"]
        ckpt_dir  = ds_cfg["checkpoint_dir"]

        if not Path(test_dir).exists():
            print(f"\n[{ds_name}] Test directory not found: {test_dir}")
            print(f"  Skipping {ds_name}.")
            continue

        print(f"\n{'=' * 70}")
        print(f"Evaluating on {ds_name}")
        print(f"{'=' * 70}")
        print(f"  Test dir     : {test_dir}")
        print(f"  Checkpoints  : {ckpt_dir}")

        test_images, test_labels = load_images_and_labels(test_dir)
        if len(test_images) == 0:
            print(f"  No test images found in {test_dir}")
            continue

        print(f"  Test images  : {len(test_images)} "
              f"(no_cancer={int((test_labels==0).sum())}, "
              f"cancer={int((test_labels==1).sum())})")

        ds_results = {}

        # Base models
        for model_name in MODELS:
            print(f"\n  [{model_name}]")
            preds, proba = get_model_predictions(
                model_name, ckpt_dir, test_images, test_labels)
            if preds is None:
                print(f"    Skipped (no checkpoint)")
                continue
            metrics = compute_metrics(test_labels, preds, proba)
            ds_results[model_name] = metrics
            print(f"    Accuracy={metrics['accuracy']:.1f}%  "
                  f"Recall={metrics['recall']:.1f}%  "
                  f"AUC={metrics['roc_auc']:.1f}%")

        # Ensemble
        print(f"\n  [ensemble]")
        ens_preds, ens_proba = get_ensemble_predictions(
            ckpt_dir, test_images, test_labels)
        if ens_preds is not None:
            metrics = compute_metrics(test_labels, ens_preds, ens_proba)
            ds_results["ensemble"] = metrics
            print(f"    Accuracy={metrics['accuracy']:.1f}%  "
                  f"Recall={metrics['recall']:.1f}%  "
                  f"AUC={metrics['roc_auc']:.1f}%")

        results[ds_name] = ds_results
        available_datasets.append(ds_name)

    if len(available_datasets) == 0:
        print("\nNo datasets available for comparison.")
        print("Make sure you have trained models on at least one dataset.")
        sys.exit(1)

    # ---- Print table ----
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print_comparison_table(results)

    # ---- Save outputs ----
    print(f"\n{'=' * 70}")
    print("Saving outputs...")

    if len(available_datasets) >= 2:
        plot_accuracy_comparison(results, f"{OUT_DIR}/accuracy_comparison.png")
        plot_auc_comparison(results,      f"{OUT_DIR}/auc_comparison.png")
        plot_metrics_heatmap(results,     f"{OUT_DIR}/full_metrics_heatmap.png")
    else:
        print(f"  Only {available_datasets[0]} available — "
              "comparison plots require both datasets.")

    save_csv(results,          f"{OUT_DIR}/comparison_table.csv")
    save_text_summary(results, f"{OUT_DIR}/cross_dataset_summary.txt")

    print(f"\nAll outputs saved to: {OUT_DIR}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
