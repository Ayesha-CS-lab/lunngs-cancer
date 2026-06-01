"""
Manual Kaggle Dataset Integration Script
==========================================

Use this when you've manually downloaded the dataset from Kaggle.

Usage:
    python prepare_kaggle_dataset_manual.py --source "C:\path\to\dataset"

The source folder should contain: Normal/, Benign/, Malignant/
"""

import argparse
import shutil
from pathlib import Path
import random
from tqdm import tqdm


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def analyze_dataset(source_path):
    """Analyze dataset structure."""
    print("\n" + "="*70)
    print("Analyzing Dataset Structure")
    print("="*70 + "\n")
    
    source_path = Path(source_path)
    
    if not source_path.exists():
        print(f"❌ Error: Path does not exist: {source_path}")
        return None
    
    print(f"Source path: {source_path}\n")
    
    # Count images - handle different folder naming conventions
    categories = {}
    folder_mapping = {
        "Normal": ["Normal", "Normal cases"],
        "Benign": ["Benign", "Bengin cases", "Benign cases"],  # Handle typo
        "Malignant": ["Malignant", "Malignant cases"]
    }
    
    for category, possible_names in folder_mapping.items():
        found = False
        for folder_name in possible_names:
            category_path = source_path / folder_name
            if category_path.exists():
                images = list(category_path.glob("*.png")) + \
                        list(category_path.glob("*.jpg")) + \
                        list(category_path.glob("*.jpeg"))
                categories[category] = len(images)
                print(f"✓ {category:12} : {len(images):4d} images (from '{folder_name}')")
                found = True
                break
        
        if not found:
            print(f"❌ {category:12} : Folder not found")
            categories[category] = 0
    
    total = sum(categories.values())
    print(f"\n{'Total':12} : {total:4d} images")
    
    if total == 0:
        print("\n❌ No images found. Please check the source path.")
        return None
    
    return categories


def reorganize_dataset(source_path, target_base="data/kaggle_lung_cancer"):
    """Reorganize into binary classification."""
    print("\n" + "="*70)
    print("Reorganizing for Binary Classification")
    print("="*70 + "\n")
    
    print("Strategy: Normal + Benign → no_cancer, Malignant → cancer\n")
    
    source_path = Path(source_path)
    
    # Create directories
    ensure_dir(f"{target_base}/raw/no_cancer")
    ensure_dir(f"{target_base}/raw/cancer")
    
    # Copy Normal (handle different folder names)
    print("Copying Normal images...")
    for folder_name in ["Normal", "Normal cases"]:
        normal_path = source_path / folder_name
        if normal_path.exists():
            images = list(normal_path.glob("*.png")) + \
                    list(normal_path.glob("*.jpg")) + \
                    list(normal_path.glob("*.jpeg"))
            for img in tqdm(images, desc="Normal"):
                shutil.copy2(img, f"{target_base}/raw/no_cancer/normal_{img.name}")
            break
    
    # Copy Benign (handle typo: "Bengin cases")
    print("Copying Benign images...")
    for folder_name in ["Benign", "Bengin cases", "Benign cases"]:
        benign_path = source_path / folder_name
        if benign_path.exists():
            images = list(benign_path.glob("*.png")) + \
                    list(benign_path.glob("*.jpg")) + \
                    list(benign_path.glob("*.jpeg"))
            for img in tqdm(images, desc="Benign"):
                shutil.copy2(img, f"{target_base}/raw/no_cancer/benign_{img.name}")
            break
    
    # Copy Malignant
    print("Copying Malignant images...")
    for folder_name in ["Malignant", "Malignant cases"]:
        malignant_path = source_path / folder_name
        if malignant_path.exists():
            images = list(malignant_path.glob("*.png")) + \
                    list(malignant_path.glob("*.jpg")) + \
                    list(malignant_path.glob("*.jpeg"))
            for img in tqdm(images, desc="Malignant"):
                shutil.copy2(img, f"{target_base}/raw/cancer/malignant_{img.name}")
            break
    
    # Summary
    no_cancer_count = len(list(Path(f"{target_base}/raw/no_cancer").glob("*")))
    cancer_count = len(list(Path(f"{target_base}/raw/cancer").glob("*")))
    
    print(f"\n✓ Reorganization complete!")
    print(f"\n  no_cancer : {no_cancer_count:4d} images")
    print(f"  cancer    : {cancer_count:4d} images")
    print(f"  Total     : {no_cancer_count + cancer_count:4d} images")
    
    return target_base


def split_dataset(data_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split into train/val/test."""
    print("\n" + "="*70)
    print("Splitting into Train/Val/Test")
    print("="*70 + "\n")
    
    set_seed(42)
    data_path = Path(data_path)
    
    for class_name in ["no_cancer", "cancer"]:
        print(f"Processing {class_name}...")
        
        source_dir = data_path / "raw" / class_name
        images = list(source_dir.glob("*"))
        random.shuffle(images)
        
        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]
        
        # Create directories
        ensure_dir(data_path / "train" / class_name)
        ensure_dir(data_path / "val" / class_name)
        ensure_dir(data_path / "test" / class_name)
        
        # Copy files
        for img in tqdm(train_images, desc=f"  Train {class_name}"):
            shutil.copy2(img, data_path / "train" / class_name / img.name)
        
        for img in tqdm(val_images, desc=f"  Val {class_name}"):
            shutil.copy2(img, data_path / "val" / class_name / img.name)
        
        for img in tqdm(test_images, desc=f"  Test {class_name}"):
            shutil.copy2(img, data_path / "test" / class_name / img.name)
        
        print(f"  ✓ {len(train_images)} train, {len(val_images)} val, {len(test_images)} test\n")
    
    print("="*70)
    print("Split Summary:")
    print("="*70 + "\n")
    
    for split in ["train", "val", "test"]:
        split_path = data_path / split
        no_cancer = len(list((split_path / "no_cancer").glob("*")))
        cancer = len(list((split_path / "cancer").glob("*")))
        print(f"{split.upper():5} : {no_cancer + cancer:4d} images (no_cancer: {no_cancer}, cancer: {cancer})")
    
    print(f"\n✓ Dataset ready at: {data_path.absolute()}")
    return str(data_path)


def main():
    parser = argparse.ArgumentParser(description="Prepare manually downloaded Kaggle dataset")
    parser.add_argument("--source", required=True, help="Path to extracted dataset folder containing Normal/, Benign/, Malignant/")
    parser.add_argument("--output", default="data/kaggle_lung_cancer", help="Output directory")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("MANUAL DATASET PREPARATION")
    print("="*70)
    
    # Analyze
    categories = analyze_dataset(args.source)
    if categories is None:
        return
    
    # Reorganize
    target_path = reorganize_dataset(args.source, args.output)
    
    # Split
    final_path = split_dataset(target_path)
    
    print("\n" + "="*70)
    print("✅ DATASET PREPARATION COMPLETE!")
    print("="*70 + "\n")
    
    print("Next steps:")
    print(f"\n1. Data ready at: {final_path}")
    print(f"\n2. Install ML dependencies:")
    print(f"   pip install torch torchvision opencv-python numpy scikit-learn")
    print(f"\n3. Train models:")
    print(f"   python train.py --model efficientnet --data_dir {final_path}")


if __name__ == "__main__":
    main()
