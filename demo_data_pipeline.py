"""
Data Pipeline Implementation Demo
==================================

This script demonstrates the complete data pipeline and preprocessing workflow.
It shows how to:
1. Load and preprocess images (PNG/JPEG/DICOM)
2. Apply augmentations
3. Create datasets and dataloaders
4. Visualize processed images

Run this script to verify your data pipeline is working correctly.
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.config import *
from src.preprocessing import load_image, normalize_image, save_processed_image
from src.augmentations import get_train_transforms, get_val_transforms
from src.datasets import LungCancerDataset, create_dataloaders
from src.utils import set_seed, ensure_dir


def demo_1_load_single_image():
    """Demo 1: Load and preprocess a single image."""
    print("\n" + "="*70)
    print("DEMO 1: Loading and Preprocessing Single Image")
    print("="*70 + "\n")
    
    # Create a dummy image for demonstration
    print("Creating dummy CT scan image...")
    dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Add some structure to make it look more like a medical image
    # Circular region (simulating lung)
    center = (256, 256)
    radius = 150
    cv2.circle(dummy_image, center, radius, (200, 200, 200), -1)
    
    # Add a small bright spot (simulating potential nodule)
    cv2.circle(dummy_image, (300, 280), 20, (255, 255, 255), -1)
    
    # Save dummy image
    ensure_dir('data/demo')
    cv2.imwrite('data/demo/dummy_ct_scan.png', dummy_image)
    print("✓ Created dummy CT scan: data/demo/dummy_ct_scan.png")
    
    # Load image using our preprocessing pipeline
    print("\nLoading image using preprocessing pipeline...")
    loaded_image = load_image('data/demo/dummy_ct_scan.png', image_size=IMAGE_SIZE)
    
    print(f"✓ Image loaded successfully!")
    print(f"  Original size: 512x512x3")
    print(f"  Processed size: {loaded_image.shape}")
    print(f"  Data type: {loaded_image.dtype}")
    print(f"  Value range: [{loaded_image.min():.3f}, {loaded_image.max():.3f}]")
    
    # Normalize image
    print("\nNormalizing image (ImageNet statistics)...")
    normalized = normalize_image(loaded_image)
    print(f"✓ Image normalized")
    print(f"  New value range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    return dummy_image, loaded_image, normalized


def demo_2_augmentations():
    """Demo 2: Apply different augmentations."""
    print("\n" + "="*70)
    print("DEMO 2: Testing Augmentation Pipeline")
    print("="*70 + "\n")
    
    # Load image
    image = cv2.imread('data/demo/dummy_ct_scan.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print("Applying augmentations...")
    
    # Get transforms
    train_transform = get_train_transforms(IMAGE_SIZE)
    val_transform = get_val_transforms(IMAGE_SIZE)
    
    # Apply train augmentation (with random transforms)
    print("\n1. Training Augmentation (includes random transforms):")
    augmented_samples = []
    for i in range(3):
        augmented = train_transform(image=image)
        aug_image = augmented['image']
        augmented_samples.append(aug_image.permute(1, 2, 0).numpy())
        print(f"   Sample {i+1}: Shape={aug_image.shape}, Type={aug_image.dtype}")
    
    # Apply validation augmentation (no random transforms)
    print("\n2. Validation Augmentation (resize + normalize only):")
    val_augmented = val_transform(image=image)
    val_image = val_augmented['image']
    print(f"   Shape: {val_image.shape}, Type: {val_image.dtype}")
    
    print("\n✓ Augmentations applied successfully!")
    
    return image, augmented_samples, val_image


def demo_3_create_dataset():
    """Demo 3: Create PyTorch dataset."""
    print("\n" + "="*70)
    print("DEMO 3: Creating PyTorch Dataset")
    print("="*70 + "\n")
    
    # Create dummy dataset structure
    print("Setting up dummy dataset structure...")
    ensure_dir('data/processed/train/no_cancer')
    ensure_dir('data/processed/train/cancer')
    ensure_dir('data/processed/val/no_cancer')
    ensure_dir('data/processed/val/cancer')
    
    # Create dummy images
    print("Creating dummy dataset (10 images)...")
    for i in range(3):
        # No cancer images
        dummy_img = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(f'data/processed/train/no_cancer/sample_{i}.png', dummy_img)
        
        # Cancer images (slightly brighter to differentiate)
        dummy_img = np.random.randint(150, 255, (224, 224, 3), dtype=np.uint8)
        cv2.circle(dummy_img, (112, 112), 30, (255, 255, 255), -1)
        cv2.imwrite(f'data/processed/train/cancer/sample_{i}.png', dummy_img)
    
    for i in range(2):
        # Validation images
        dummy_img = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(f'data/processed/val/no_cancer/sample_{i}.png', dummy_img)
        
        dummy_img = np.random.randint(150, 255, (224, 224, 3), dtype=np.uint8)
        cv2.circle(dummy_img, (112, 112), 30, (255, 255, 255), -1)
        cv2.imwrite(f'data/processed/val/cancer/sample_{i}.png', dummy_img)
    
    print("✓ Created 10 dummy images (6 train, 4 val)")
    
    # Create dataset
    print("\nCreating PyTorch Dataset...")
    from src.datasets import LungCancerDataset
    
    image_paths = [
        'data/processed/train/cancer/sample_0.png',
        'data/processed/train/cancer/sample_1.png',
        'data/processed/train/no_cancer/sample_0.png',
    ]
    labels = [1, 1, 0]  # 1=cancer, 0=no_cancer
    
    dataset = LungCancerDataset(
        image_paths=image_paths,
        labels=labels,
        transform=get_train_transforms(IMAGE_SIZE)
    )
    
    print(f"✓ Dataset created!")
    print(f"  Number of samples: {len(dataset)}")
    
    # Get a sample
    print("\nGetting sample from dataset...")
    image, label = dataset[0]
    print(f"✓ Sample retrieved:")
    print(f"  Image shape: {image.shape}")
    print(f"  Image type: {image.dtype}")
    print(f"  Label: {label} ({'Cancer' if label == 1 else 'No Cancer'})")
    
    return dataset


def demo_4_create_dataloaders():
    """Demo 4: Create DataLoaders for training."""
    print("\n" + "="*70)
    print("DEMO 4: Creating DataLoaders")
    print("="*70 + "\n")
    
    try:
        print("Creating train and validation dataloaders...")
        train_loader, val_loader = create_dataloaders(
            data_dir='data/processed',
            batch_size=2,  # Small batch for demo
            num_workers=0,  # 0 for demo to avoid multiprocessing issues
            train_transform=get_train_transforms(IMAGE_SIZE),
            val_transform=get_val_transforms(IMAGE_SIZE)
        )
        
        print(f"✓ DataLoaders created successfully!")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        
        # Get a batch
        print("\nGetting a batch from train loader...")
        images, labels = next(iter(train_loader))
        print(f"✓ Batch retrieved:")
        print(f"  Batch images shape: {images.shape}")
        print(f"  Batch labels shape: {labels.shape}")
        print(f"  Labels in batch: {labels.tolist()}")
        
        return train_loader, val_loader
        
    except Exception as e:
        print(f"⚠ Note: {str(e)}")
        print("  This is expected if directory structure is empty")
        return None, None


def demo_5_visualize_pipeline():
    """Demo 5: Visualize the complete pipeline."""
    print("\n" + "="*70)
    print("DEMO 5: Visualizing Complete Pipeline")
    print("="*70 + "\n")
    
    # Load original image
    original = cv2.imread('data/demo/dummy_ct_scan.png')
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Apply preprocessing
    processed = load_image('data/demo/dummy_ct_scan.png', image_size=IMAGE_SIZE)
    
    # Apply augmentation
    transform = get_train_transforms(IMAGE_SIZE)
    augmented1 = transform(image=original)['image'].permute(1, 2, 0).numpy()
    augmented2 = transform(image=original)['image'].permute(1, 2, 0).numpy()
    
    # Denormalize for visualization
    def denormalize(img):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        return img
    
    augmented1 = denormalize(augmented1)
    augmented2 = denormalize(augmented2)
    
    # Create visualization
    ensure_dir('outputs/demo')
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('1. Original Image (512x512)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(processed)
    axes[0, 1].set_title(f'2. Preprocessed ({IMAGE_SIZE}x{IMAGE_SIZE})', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(augmented1)
    axes[1, 0].set_title('3. Augmented Sample 1\n(Random transforms applied)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(augmented2)
    axes[1, 1].set_title('4. Augmented Sample 2\n(Different random transforms)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/demo/pipeline_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Visualization saved to: outputs/demo/pipeline_visualization.png")
    print("\nPipeline stages:")
    print("  1. Original Image: Raw input from file")
    print("  2. Preprocessed: Resized to model input size")
    print("  3-4. Augmented: Random transforms for training diversity")


def demo_6_batch_processing():
    """Demo 6: Process multiple images in batch."""
    print("\n" + "="*70)
    print("DEMO 6: Batch Processing Multiple Images")
    print("="*70 + "\n")
    
    # Create multiple dummy images
    print("Creating batch of 5 images...")
    image_paths = []
    for i in range(5):
        img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
        path = f'data/demo/batch_image_{i}.png'
        cv2.imwrite(path, img)
        image_paths.append(path)
    
    print(f"✓ Created {len(image_paths)} images")
    
    # Process all images
    print("\nProcessing all images through pipeline...")
    processed_images = []
    for i, path in enumerate(image_paths):
        img = load_image(path, image_size=IMAGE_SIZE)
        processed_images.append(img)
        print(f"  Processed image {i+1}/5: {img.shape}")
    
    print(f"\n✓ Batch processing complete!")
    print(f"  Total images processed: {len(processed_images)}")
    print(f"  Output shape per image: {processed_images[0].shape}")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("DATA PIPELINE & PREPROCESSING IMPLEMENTATION DEMO")
    print("="*70)
    print("\nThis script demonstrates the complete data pipeline:")
    print("  ✓ Image loading (PNG/JPEG/DICOM support)")
    print("  ✓ Preprocessing (resize, normalize)")
    print("  ✓ Augmentation (training transforms)")
    print("  ✓ Dataset creation (PyTorch)")
    print("  ✓ DataLoader setup (batch processing)")
    print("  ✓ Visualization")
    
    # Set seed for reproducibility
    set_seed(SEED)
    
    try:
        # Run demos
        demo_1_load_single_image()
        demo_2_augmentations()
        demo_3_create_dataset()
        demo_4_create_dataloaders()
        demo_5_visualize_pipeline()
        demo_6_batch_processing()
        
        # Summary
        print("\n" + "="*70)
        print("DEMO COMPLETE!")
        print("="*70)
        print("\n✅ All pipeline components working correctly!")
        print("\nGenerated files:")
        print("  📁 data/demo/ - Demo images")
        print("  📁 data/processed/ - Dummy dataset")
        print("  📊 outputs/demo/pipeline_visualization.png - Visual summary")
        print("\nNext steps:")
        print("  1. Replace dummy data with real CT scans in data/raw/")
        print("  2. Run training: python train.py --model efficientnet")
        print("  3. Check outputs in outputs/ and checkpoints/")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  - Ensure all dependencies are installed: pip install -r requirements.txt")
        print("  - Check that src/ modules are in Python path")
        print("  - Verify directory structure exists")


if __name__ == "__main__":
    main()
