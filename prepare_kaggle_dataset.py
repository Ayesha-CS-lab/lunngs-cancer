"""
Kaggle Dataset Integration Script (Minimal Version)
====================================================

Downloads and prepares the IQ-OTHNCCD Lung Cancer Dataset from Kaggle.
This version only requires: kagglehub, Pillow, tqdm
"""

import os
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

def download_kaggle_dataset():
    """Download dataset from Kaggle."""
    print("\n" + "="*70)
    print("STEP 1: Downloading Kaggle Dataset")
    print("="*70 + "\n")
    
    try:
        import kagglehub
        
        print("Downloading IQ-OTHNCCD Lung Cancer Dataset...")
        print("This may take several minutes depending on your connection...\n")
        
        # Download dataset
        path = kagglehub.dataset_download("adityamahimkar/iqothnccd-lung-cancer-dataset")
        
        print(f"\n✓ Dataset downloaded to: {path}")
        
        return path
        
    except ImportError:
        print("❌ Error: kagglehub not installed")
        print("\nInstall with:")
        print("  pip install kagglehub")
        return None
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure you have Kaggle API credentials configured")
        print("  2. Run: kaggle datasets download -d adityamahimkar/iqothnccd-lung-cancer-dataset")
        print("  3. Or download manually from Kaggle website")
        return None


def analyze_dataset_structure(download_path):
    """Analyze the downloaded dataset structure."""
    print("\n" + "="*70)
    print("STEP 2: Analyzing Dataset Structure")
    print("="*70 + "\n")
    
    dataset_path = Path(download_path)
    
    # Find the actual dataset folder
    possible_paths = [
        dataset_path,
        dataset_path / "iqothnccd-lung-cancer-dataset",
        dataset_path / "lung-cancer-dataset"
    ]
    
    actual_path = None
    for path in possible_paths:
        if path.exists() and (path / "Normal").exists():
            actual_path = path
            break
    
    if actual_path is None:
        print("⚠️  Could not find dataset folders")
        print(f"   Searching in: {dataset_path}")
        print("\n   Available folders:")
        for item in dataset_path.rglob("*"):
            if item.is_dir():
                print(f"     - {item.relative_to(dataset_path)}")
        return None
    
    print(f"Dataset location: {actual_path}\n")
    
    # Count images in each category
    categories = {}
    for category in ["Normal", "Benign", "Malignant"]:
        category_path = actual_path / category
        if category_path.exists():
            image_files = list(category_path.glob("*.png")) + \
                         list(category_path.glob("*.jpg")) + \
                         list(category_path.glob("*.jpeg"))
            categories[category] = len(image_files)
            print(f"✓ {category:12} : {len(image_files):4d} images")
        else:
            print(f"❌ {category:12} : Folder not found")
            categories[category] = 0
    
    total = sum(categories.values())
    print(f"\n{'Total':12} : {total:4d} images")
    
    # Class distribution
    print("\nClass Distribution:")
    for category, count in categories.items():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  {category:12} : {percentage:5.1f}%")
    
    return actual_path, categories


def reorganize_for_binary_classification(source_path, target_base="data/kaggle_lung_cancer"):
    """Reorganize 3-class dataset into binary classification."""
    print("\n" + "="*70)
    print("STEP 3: Reorganizing for Binary Classification")
    print("="*70 + "\n")
    
    print("Classification strategy:")
    print("  • Normal + Benign → no_cancer")
    print("  • Malignant → cancer")
    print("\nThis focuses on detecting malignant cancer specifically.\n")
    
    # Create target structure
    ensure_dir(f"{target_base}/raw/no_cancer")
    ensure_dir(f"{target_base}/raw/cancer")
    
    source_path = Path(source_path)
    
    # Copy Normal to no_cancer
    print("Copying Normal images to no_cancer...")
    normal_path = source_path / "Normal"
    if normal_path.exists():
        images = list(normal_path.glob("*.png")) + \
                list(normal_path.glob("*.jpg")) + \
                list(normal_path.glob("*.jpeg"))
        
        for img_path in tqdm(images, desc="Normal"):
            target_name = f"normal_{img_path.name}"
            shutil.copy2(img_path, f"{target_base}/raw/no_cancer/{target_name}")
    
    # Copy Benign to no_cancer
    print("Copying Benign images to no_cancer...")
    benign_path = source_path / "Benign"
    if benign_path.exists():
        images = list(benign_path.glob("*.png")) + \
                list(benign_path.glob("*.jpg")) + \
                list(benign_path.glob("*.jpeg"))
        
        for img_path in tqdm(images, desc="Benign"):
            target_name = f"benign_{img_path.name}"
            shutil.copy2(img_path, f"{target_base}/raw/no_cancer/{target_name}")
    
    # Copy Malignant to cancer
    print("Copying Malignant images to cancer...")
    malignant_path = source_path / "Malignant"
    if malignant_path.exists():
        images = list(malignant_path.glob("*.png")) + \
                list(malignant_path.glob("*.jpg")) + \
                list(malignant_path.glob("*.jpeg"))
        
        for img_path in tqdm(images, desc="Malignant"):
            target_name = f"malignant_{img_path.name}"
            shutil.copy2(img_path, f"{target_base}/raw/cancer/{target_name}")
    
    # Count final distribution
    no_cancer_count = len(list(Path(f"{target_base}/raw/no_cancer").glob("*")))
    cancer_count = len(list(Path(f"{target_base}/raw/cancer").glob("*")))
    
    print(f"\n✓ Reorganization complete!")
    print(f"\nBinary classification distribution:")
    print(f"  no_cancer : {no_cancer_count:4d} images (Normal + Benign)")
    print(f"  cancer    : {cancer_count:4d} images (Malignant)")
    
    total = no_cancer_count + cancer_count
    print(f"\n  Total     : {total:4d} images")
    
    # Check imbalance
    if cancer_count > 0:
        ratio = no_cancer_count / cancer_count
        print(f"\n  Imbalance ratio: {ratio:.2f}:1 (no_cancer:cancer)")
        
        if ratio > 3:
            print("\n  ⚠️  Significant class imbalance detected!")
            print("     Recommendations:")
            print("     1. Use class weights in training")
            print("     2. Consider GAN augmentation for minority class")
            print("     3. Use stratified splitting")
    
    return target_base


def split_dataset(data_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split dataset into train/val/test sets."""
    print("\n" + "="*70)
    print("STEP 4: Splitting into Train/Val/Test Sets")
    print("="*70 + "\n")
    
    set_seed(seed)
    
    print(f"Split ratios: Train={train_ratio:.0%}, Val={val_ratio:.0%}, Test={test_ratio:.0%}\n")
    
    data_path = Path(data_path)
    
    for class_name in ["no_cancer", "cancer"]:
        print(f"Processing {class_name}...")
        
        # Get all images
        source_dir = data_path / "raw" / class_name
        images = list(source_dir.glob("*"))
        
        # Shuffle
        random.shuffle(images)
        
        # Calculate split indices
        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        # Split
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
        
        print(f"  ✓ {class_name}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test\n")
    
    # Summary
    print("="*70)
    print("Split Summary:")
    print("="*70 + "\n")
    
    for split in ["train", "val", "test"]:
        split_path = data_path / split
        no_cancer = len(list((split_path / "no_cancer").glob("*")))
        cancer = len(list((split_path / "cancer").glob("*")))
        total = no_cancer + cancer
        
        print(f"{split.upper():5} : {total:4d} images (no_cancer: {no_cancer:4d}, cancer: {cancer:4d})")
    
    print(f"\n✓ Dataset split complete!")
    print(f"\nData ready at: {data_path.absolute()}")
    
    return str(data_path)


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("KAGGLE LUNG CANCER DATASET INTEGRATION")
    print("="*70)
    print("\nThis script will:")
    print("  1. Download dataset from Kaggle")
    print("  2. Analyze dataset structure")
    print("  3. Reorganize for binary classification")
    print("  4. Split into train/val/test sets")
    print("  5. Prepare for model training")
    print("\nMake sure you have:")
    print("  • Kaggle account and API credentials configured")
    print("  • kagglehub installed (pip install kagglehub)")
    print("  • Sufficient disk space (~500MB-1GB)")
    
    input("\nPress Enter to continue...")
    
    try:
        # Step 1: Download
        download_path = download_kaggle_dataset()
        if download_path is None:
            print("\n❌ Download failed. Exiting.")
            return
        
        # Step 2: Analyze
        result = analyze_dataset_structure(download_path)
        if result is None:
            print("\n❌ Could not find dataset. Exiting.")
            return
        dataset_path, categories = result
        
        # Step 3: Reorganize
        target_path = reorganize_for_binary_classification(dataset_path)
        
        # Step 4: Split
        final_path = split_dataset(target_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        
        # Final summary
        print("\n" + "="*70)
        print("✅ DATASET INTEGRATION COMPLETE!")
        print("="*70 + "\n")
        
        print("Next steps:")
        print(f"\n1. Data is ready at: {final_path}")
        print(f"\n2. Install remaining dependencies:")
        print(f"   pip install torch torchvision opencv-python numpy scikit-learn albumentations")
        print(f"\n3. Train models:")
        print(f"   python train.py --model efficientnet --epochs 50 --data_dir {final_path}")
        print(f"\n4. Evaluate:")
        print(f"   python evaluate.py --model efficientnet")
        print(f"\n5. Launch demo app:")
        print(f"   python demo_app.py")
        
        # Data statistics
        print(f"\n{'='*70}")
        print("Dataset Statistics:")
        print(f"{'='*70}\n")
        
        train_path = Path(final_path) / "train"
        val_path = Path(final_path) / "val"
        test_path = Path(final_path) / "test"
        
        for split_path, split_name in [(train_path, "Training"), 
                                       (val_path, "Validation"), 
                                       (test_path, "Testing")]:
            if split_path.exists():
                no_cancer = len(list((split_path / "no_cancer").glob("*")))
                cancer = len(list((split_path / "cancer").glob("*")))
                print(f"{split_name:12} : {no_cancer + cancer:4d} images")
                print(f"  └─ no_cancer: {no_cancer:4d}")
                print(f"  └─ cancer:    {cancer:4d}\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  1. Ensure kagglehub is installed: pip install kagglehub")
        print("  2. Configure Kaggle API: https://www.kaggle.com/docs/api")
        print("  3. Check disk space and permissions")


if __name__ == "__main__":
    main()
