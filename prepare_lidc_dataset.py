"""
LIDC-IDRI Dataset Preparation Script
======================================
Downloads the LIDC-IDRI dataset from Kaggle, converts it to the same
binary classification folder structure used by IQ-OTH/NCCD, then splits
into train / val / test sets.

Binary classification rule (standard in literature):
  Malignancy score >= 3  →  cancer
  Malignancy score <  3  →  no_cancer
  (Score 3 is ambiguous; by default it is included as cancer — see --exclude-ambiguous)

Supported Kaggle dataset formats (auto-detected):
  A) CSV + image folder  — CSV has a 'malignancy' column, images in one flat dir
  B) Malignancy sub-folders — data/<1..5>/ or data/malignancy_<1..5>/
  C) Pre-split binary folders — data/{cancer,no_cancer}/ or data/{malignant,benign}/

Usage on Kaggle notebook (add dataset as input first):
    python prepare_lidc_dataset.py --source-dir /kaggle/input/<dataset-slug>

Usage on local machine:
    python prepare_lidc_dataset.py --download
    python prepare_lidc_dataset.py --source-dir /path/to/lidc/data
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

OUTPUT_BASE = "data/lidc_idri"
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SEED        = 42

KAGGLE_SLUG = "nicholasgunning/lidc-idri-preprocessed"  # adjust if needed


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def count_split(base, split):
    no_c = len(list((Path(base) / split / "no_cancer").glob("*")))
    can  = len(list((Path(base) / split / "cancer").glob("*")))
    return no_c, can


def copy_image(src, dst):
    """Copy src to dst, skipping unreadable files."""
    try:
        img = cv2.imread(str(src))
        if img is None:
            return False
        shutil.copy2(str(src), str(dst))
        return True
    except Exception:
        return False


def split_and_copy(image_paths, labels, output_base, seed=42):
    """
    Stratified split into train/val/test and copy files.
    image_paths: list of Path objects
    labels: list of 'cancer' or 'no_cancer' strings
    """
    random.seed(seed)
    cancer_paths    = [p for p, l in zip(image_paths, labels) if l == "cancer"]
    no_cancer_paths = [p for p, l in zip(image_paths, labels) if l == "no_cancer"]

    for class_name, paths in [("cancer", cancer_paths), ("no_cancer", no_cancer_paths)]:
        random.shuffle(paths)
        n      = len(paths)
        t_end  = int(n * TRAIN_RATIO)
        v_end  = t_end + int(n * VAL_RATIO)

        splits = {
            "train": paths[:t_end],
            "val":   paths[t_end:v_end],
            "test":  paths[v_end:],
        }

        for split_name, split_paths in splits.items():
            dst_dir = Path(output_base) / split_name / class_name
            ensure_dir(dst_dir)
            ok = 0
            for src in tqdm(split_paths, desc=f"  {split_name}/{class_name}"):
                dst = dst_dir / f"lidc_{src.stem}{src.suffix}"
                if copy_image(src, dst):
                    ok += 1
            print(f"    {split_name}/{class_name}: {ok}/{len(split_paths)} images copied")


# ---------------------------------------------------------------------------
# Format detectors
# ---------------------------------------------------------------------------

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def collect_images(directory):
    return sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in IMG_EXTS)


def detect_format(source_dir):
    """Return one of: 'csv', 'score_folders', 'binary_folders', 'unknown'."""
    source_dir = Path(source_dir)

    # A) CSV + images
    csv_files = list(source_dir.rglob("*.csv"))
    if csv_files:
        return "csv", csv_files[0]

    # B) Score sub-folders (1..5)
    score_dirs = [d for d in source_dir.rglob("*")
                  if d.is_dir() and d.name.lstrip("malignancy_").lstrip("score_") in
                  {"1", "2", "3", "4", "5"}]
    if len(score_dirs) >= 2:
        return "score_folders", None

    # C) Pre-split binary folders
    binary_names = {"cancer", "no_cancer", "malignant", "benign", "normal",
                    "positive", "negative"}
    binary_dirs = [d for d in source_dir.rglob("*")
                   if d.is_dir() and d.name.lower() in binary_names]
    if binary_dirs:
        return "binary_folders", None

    return "unknown", None


# ---------------------------------------------------------------------------
# Format handlers
# ---------------------------------------------------------------------------

def handle_csv_format(source_dir, csv_path, exclude_ambiguous):
    import csv
    print(f"  Using CSV format — {csv_path.name}")

    source_dir = Path(source_dir)
    image_paths = []
    labels      = []
    skipped     = 0

    # Find image directory (largest image collection sibling of CSV)
    img_dirs = sorted(
        [d for d in source_dir.rglob("*") if d.is_dir()],
        key=lambda d: len(collect_images(d)),
        reverse=True
    )
    img_dir = img_dirs[0] if img_dirs else source_dir
    img_lookup = {p.stem: p for p in collect_images(img_dir)}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fields = [k.lower().strip() for k in reader.fieldnames] if reader.fieldnames else []

        # Try to find the malignancy column
        mal_col = next(
            (orig for orig, low in zip(reader.fieldnames, fields)
             if "malig" in low or "label" in low or "class" in low),
            None
        )
        # Try to find the image ID / filename column
        id_col  = next(
            (orig for orig, low in zip(reader.fieldnames, fields)
             if "id" in low or "file" in low or "name" in low or "image" in low),
            None
        )

        if mal_col is None:
            print("  Cannot find malignancy column in CSV. Columns:", reader.fieldnames)
            return [], []

        f.seek(0)
        next(reader)  # skip header again

        for row in reader:
            try:
                score = float(row[mal_col])
            except (ValueError, KeyError):
                skipped += 1
                continue

            # Ambiguous handling
            if exclude_ambiguous and score == 3:
                skipped += 1
                continue

            label = "cancer" if score >= 3 else "no_cancer"

            # Locate image
            img_path = None
            if id_col:
                stem = str(row[id_col]).strip().replace(".png", "").replace(".jpg", "")
                img_path = img_lookup.get(stem)
            if img_path is None:
                # Try all images — match any key containing the row index
                pass

            if img_path is not None:
                image_paths.append(img_path)
                labels.append(label)
            else:
                skipped += 1

    print(f"  Matched: {len(image_paths)} | Skipped: {skipped}")
    return image_paths, labels


def handle_score_folders(source_dir, exclude_ambiguous):
    """Handles folders named 1, 2, 3, 4, 5 (or malignancy_1 etc.)."""
    source_dir = Path(source_dir)
    image_paths = []
    labels      = []

    for d in sorted(source_dir.rglob("*")):
        if not d.is_dir():
            continue
        raw = d.name.lstrip("malignancy_").lstrip("score_").lstrip("Malignancy_")
        if raw not in {"1", "2", "3", "4", "5"}:
            continue
        score = int(raw)

        if exclude_ambiguous and score == 3:
            print(f"  Skipping ambiguous folder: {d.name}")
            continue

        label = "cancer" if score >= 3 else "no_cancer"
        imgs  = collect_images(d)
        print(f"  Folder '{d.name}' (score={score}) → {label}: {len(imgs)} images")
        for img in imgs:
            image_paths.append(img)
            labels.append(label)

    return image_paths, labels


def handle_binary_folders(source_dir):
    """Handles pre-labelled cancer / no_cancer / malignant / benign folders."""
    source_dir = Path(source_dir)
    image_paths = []
    labels      = []

    CANCER_NAMES    = {"cancer", "malignant", "positive"}
    NO_CANCER_NAMES = {"no_cancer", "benign", "normal", "negative"}

    for d in source_dir.rglob("*"):
        if not d.is_dir():
            continue
        name = d.name.lower()
        if name in CANCER_NAMES:
            label = "cancer"
        elif name in NO_CANCER_NAMES:
            label = "no_cancer"
        else:
            continue

        imgs = collect_images(d)
        print(f"  Folder '{d.name}' → {label}: {len(imgs)} images")
        for img in imgs:
            image_paths.append(img)
            labels.append(label)

    return image_paths, labels


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_from_kaggle(slug):
    print(f"\nDownloading LIDC-IDRI dataset from Kaggle: {slug}")
    print("This may take several minutes...\n")
    try:
        import kagglehub
        path = kagglehub.dataset_download(slug)
        print(f"Downloaded to: {path}")
        return path
    except ImportError:
        print("kagglehub not installed. Run:  pip install kagglehub")
        sys.exit(1)
    except Exception as e:
        print(f"Download failed: {e}")
        print("\nAlternatives:")
        print(f"  1. Download manually from https://www.kaggle.com/datasets/{slug}")
        print(f"     then run:  python prepare_lidc_dataset.py --source-dir <unzipped_path>")
        print(f"  2. On Kaggle notebook: Add the dataset as input, then use --source-dir /kaggle/input/<slug>")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare LIDC-IDRI dataset for training")
    parser.add_argument("--source-dir",          type=str, default=None,
                        help="Path to downloaded/unzipped LIDC-IDRI data")
    parser.add_argument("--download",            action="store_true",
                        help="Download dataset from Kaggle automatically")
    parser.add_argument("--kaggle-slug",         type=str, default=KAGGLE_SLUG,
                        help=f"Kaggle dataset slug (default: {KAGGLE_SLUG})")
    parser.add_argument("--output-dir",          type=str, default=OUTPUT_BASE,
                        help="Output base directory (default: data/lidc_idri)")
    parser.add_argument("--exclude-ambiguous",   action="store_true",
                        help="Exclude malignancy score=3 (ambiguous) images")
    parser.add_argument("--train-ratio",         type=float, default=TRAIN_RATIO)
    parser.add_argument("--val-ratio",           type=float, default=VAL_RATIO)
    parser.add_argument("--test-ratio",          type=float, default=TEST_RATIO)
    parser.add_argument("--seed",                type=int,   default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 70)
    print("LIDC-IDRI DATASET PREPARATION")
    print("=" * 70)
    print(f"Malignancy threshold : score >= 3 → cancer  (standard literature rule)")
    print(f"Exclude ambiguous    : {args.exclude_ambiguous} (score == 3)")
    print(f"Output               : {args.output_dir}")
    print()

    # ---- Acquire source directory ----
    if args.source_dir:
        source_dir = Path(args.source_dir)
        if not source_dir.exists():
            print(f"ERROR: --source-dir '{source_dir}' does not exist.")
            sys.exit(1)
    elif args.download:
        source_dir = Path(download_from_kaggle(args.kaggle_slug))
    else:
        print("Provide --source-dir <path>  OR  use --download")
        print()
        print("On a Kaggle notebook:")
        print("  1. Add LIDC-IDRI dataset as input to your notebook")
        print(f"     Suggested: https://www.kaggle.com/datasets/{KAGGLE_SLUG}")
        print("  2. Run:")
        print("       python prepare_lidc_dataset.py --source-dir /kaggle/input/<dataset-name>")
        sys.exit(1)

    print(f"Source directory: {source_dir}")
    print()

    # ---- Detect format ----
    print("Step 1: Detecting dataset format...")
    fmt, extra = detect_format(source_dir)
    print(f"  Detected format: {fmt}")

    image_paths = []
    labels      = []

    if fmt == "csv":
        image_paths, labels = handle_csv_format(source_dir, extra, args.exclude_ambiguous)
    elif fmt == "score_folders":
        image_paths, labels = handle_score_folders(source_dir, args.exclude_ambiguous)
    elif fmt == "binary_folders":
        image_paths, labels = handle_binary_folders(source_dir)
    else:
        print("\nCould not detect dataset format automatically.")
        print("The script supports:")
        print("  A) A CSV file with a 'malignancy' column + image folder")
        print("  B) Sub-folders named 1, 2, 3, 4, 5 (malignancy scores)")
        print("  C) Pre-split folders named: cancer / no_cancer / malignant / benign")
        print("\nPlease check the dataset structure and adjust the script if needed.")
        sys.exit(1)

    if not image_paths:
        print("No images found after processing. Check the source directory structure.")
        sys.exit(1)

    # ---- Summary before splitting ----
    n_cancer    = labels.count("cancer")
    n_no_cancer = labels.count("no_cancer")
    total       = len(labels)

    print(f"\nStep 2: Dataset summary")
    print(f"  Total images : {total}")
    print(f"  cancer       : {n_cancer}  ({n_cancer/total*100:.1f}%)")
    print(f"  no_cancer    : {n_no_cancer}  ({n_no_cancer/total*100:.1f}%)")

    ratio = (n_no_cancer / n_cancer) if n_cancer > 0 else float("inf")
    if ratio > 2.5 or ratio < 0.4:
        print(f"\n  Class imbalance ratio: {ratio:.2f}:1 (no_cancer:cancer)")
        print("  Run GAN augmentation after preparing the dataset:")
        print("    python run_gan_augmentation.py --data-dir data/lidc_idri/train")

    # ---- Split and copy ----
    print(f"\nStep 3: Splitting  (train={args.train_ratio:.0%} / "
          f"val={args.val_ratio:.0%} / test={args.test_ratio:.0%})")
    split_and_copy(image_paths, labels, args.output_dir, seed=args.seed)

    # ---- Final report ----
    print("\n" + "=" * 70)
    print("LIDC-IDRI PREPARATION COMPLETE")
    print("=" * 70)

    for split in ["train", "val", "test"]:
        no_c, can = count_split(args.output_dir, split)
        print(f"  {split.upper():5}: no_cancer={no_c:4d}, cancer={can:4d}, total={no_c+can:4d}")

    print()
    print("Next steps:")
    print("  # (Optional) GAN augmentation on LIDC-IDRI")
    print("  python run_gan_augmentation.py --data-dir data/lidc_idri/train")
    print()
    print("  # Train all three base models on LIDC-IDRI")
    print("  python train.py --model efficientnet --data-dir data/lidc_idri "
          "--checkpoint-dir checkpoints/lidc_idri --epochs 50")
    print("  python train.py --model densenet     --data-dir data/lidc_idri "
          "--checkpoint-dir checkpoints/lidc_idri --epochs 50")
    print("  python train.py --model resnet       --data-dir data/lidc_idri "
          "--checkpoint-dir checkpoints/lidc_idri --epochs 50")
    print()
    print("  # Train ensemble on LIDC-IDRI")
    print("  python train_ensemble.py "
          "--val-dir data/lidc_idri/val --test-dir data/lidc_idri/test "
          "--checkpoint-dir checkpoints/lidc_idri "
          "--output-dir outputs/lidc_idri")
    print()
    print("  # Compare both datasets")
    print("  python compare_datasets.py")


if __name__ == "__main__":
    main()
