"""
Grad-CAM Explainability Demo
=============================

This script demonstrates Grad-CAM (Gradient-weighted Class Activation Mapping)
for visualizing what regions of CT scans influence model predictions.

Features:
- Generate heatmaps showing important regions
- Overlay heatmaps on original images
- Compare Grad-CAM across different models
- Batch processing multiple images
"""

import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.config import *
from src.models.base_models import ModelFactory
from src.explainability.gradcam import GradCAM, visualize_gradcam, get_target_layer
from src.preprocessing import load_image
from src.augmentations import get_val_transforms
from src.utils import set_seed, ensure_dir


def create_demo_images():
    """Create demo CT scan images."""
    print("\n" + "="*70)
    print("STEP 1: Creating Demo Images")
    print("="*70 + "\n")
    
    ensure_dir('data/gradcam_demo')
    
    print("Creating 5 demo CT scan images with simulated tumors...")
    
    for i in range(5):
        # Create base image
        img = np.random.randint(80, 120, (224, 224, 3), dtype=np.uint8)
        
        # Add circular region (simulating lung)
        cv2.circle(img, (112, 112), 80, (100, 100, 100), -1)
        
        # Add bright spot (simulating tumor) at different locations
        tumor_locations = [(90, 90), (130, 90), (112, 112), (90, 130), (130, 130)]
        tumor_x, tumor_y = tumor_locations[i]
        cv2.circle(img, (tumor_x, tumor_y), 15, (255, 255, 255), -1)
        
        # Save
        cv2.imwrite(f'data/gradcam_demo/ct_scan_{i+1}.png', img)
    
    print("✓ Created 5 demo images with tumors at different locations")
    print("  Images saved to: data/gradcam_demo/")
    
    return 'data/gradcam_demo'


def load_pretrained_model(model_name='efficientnet', device='cuda'):
    """Load or create a model for Grad-CAM demo."""
    print("\n" + "="*70)
    print(f"STEP 2: Loading {model_name.upper()} Model")
    print("="*70 + "\n")
    
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("⚠ CUDA not available, using CPU")
    
    print(f"Device: {device}")
    print(f"Model: {model_name}\n")
    
    # Create model
    model = ModelFactory.create_model(
        model_name=model_name,
        num_classes=NUM_CLASSES,
        pretrained=True  # Using pretrained weights for demo
    ).to(device)
    
    model.eval()
    
    print(f"✓ Model loaded and ready for Grad-CAM")
    
    return model


def generate_single_gradcam(model, model_name, image_path, device='cuda'):
    """Generate Grad-CAM for a single image."""
    print("\n" + "="*70)
    print("STEP 3: Generating Grad-CAM Heatmap")
    print("="*70 + "\n")
    
    # Load image
    image = load_image(image_path, image_size=IMAGE_SIZE)
    
    # Prepare tensor
    transform = get_val_transforms(IMAGE_SIZE)
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    print(f"Processing: {Path(image_path).name}")
    print(f"Image shape: {image_tensor.shape}\n")
    
    # Generate Grad-CAM
    print("Generating heatmap...")
    overlaid, heatmap, pred_class = visualize_gradcam(
        model=model,
        model_name=model_name,
        image=image,
        image_tensor=image_tensor,
        device=device
    )
    
    print(f"✓ Grad-CAM generated!")
    print(f"  Predicted class: {'Cancer' if pred_class == 1 else 'No Cancer'}")
    print(f"  Heatmap shows regions influencing this prediction")
    
    return overlaid, heatmap, image, pred_class


def batch_gradcam_generation(model, model_name, image_dir, device='cuda'):
    """Generate Grad-CAM for multiple images."""
    print("\n" + "="*70)
    print("STEP 4: Batch Grad-CAM Generation")
    print("="*70 + "\n")
    
    image_paths = sorted(Path(image_dir).glob('*.png'))
    
    print(f"Processing {len(image_paths)} images...\n")
    
    results = []
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] {image_path.name}")
        
        # Load and preprocess
        image = load_image(str(image_path), image_size=IMAGE_SIZE)
        transform = get_val_transforms(IMAGE_SIZE)
        image_tensor = transform(image=image)['image'].unsqueeze(0).to(device)
        
        # Generate Grad-CAM
        overlaid, heatmap, pred_class = visualize_gradcam(
            model=model,
            model_name=model_name,
            image=image,
            image_tensor=image_tensor,
            device=device
        )
        
        results.append({
            'filename': image_path.name,
            'original': image,
            'overlaid': overlaid,
            'heatmap': heatmap,
            'prediction': pred_class
        })
        
        print(f"  → Prediction: {'Cancer' if pred_class == 1 else 'No Cancer'}\n")
    
    print(f"✓ Processed {len(results)} images")
    
    return results


def visualize_comparison(results, model_name='efficientnet'):
    """Create comparison visualization."""
    print("\n" + "="*70)
    print("STEP 5: Creating Visualizations")
    print("="*70 + "\n")
    
    ensure_dir('outputs/gradcam_demo')
    
    # Create grid visualization
    n_images = len(results)
    fig, axes = plt.subplots(n_images, 3, figsize=(12, 4*n_images))
    
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        # Original image
        axes[i, 0].imshow(result['original'])
        axes[i, 0].set_title(f"{result['filename']}\nOriginal CT Scan", fontsize=10, fontweight='bold')
        axes[i, 0].axis('off')
        
        # Heatmap only
        axes[i, 1].imshow(result['heatmap'], cmap='jet')
        axes[i, 1].set_title("Grad-CAM Heatmap\n(Red = High Importance)", fontsize=10, fontweight='bold')
        axes[i, 1].axis('off')
        
        # Overlay
        axes[i, 2].imshow(result['overlaid'])
        pred_text = 'Cancer' if result['prediction'] == 1 else 'No Cancer'
        axes[i, 2].set_title(f"Overlay\nPrediction: {pred_text}", fontsize=10, fontweight='bold')
        axes[i, 2].axis('off')
    
    plt.suptitle(f'Grad-CAM Explainability - {model_name.upper()}', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = f'outputs/gradcam_demo/{model_name}_gradcam_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison grid saved to: {save_path}")
    
    # Create individual overlays
    for result in results:
        filename = result['filename'].replace('.png', '_gradcam.png')
        save_path = f'outputs/gradcam_demo/{filename}'
        
        cv2.imwrite(save_path, cv2.cvtColor(result['overlaid'], cv2.COLOR_RGB2BGR))
    
    print(f"✓ Individual overlays saved to: outputs/gradcam_demo/")


def compare_models_gradcam(image_path, device='cuda'):
    """Compare Grad-CAM across different models."""
    print("\n" + "="*70)
    print("STEP 6: Comparing Grad-CAM Across Models")
    print("="*70 + "\n")
    
    model_names = ['efficientnet', 'densenet', 'resnet']
    
    # Load image
    image = load_image(image_path, image_size=IMAGE_SIZE)
    transform = get_val_transforms(IMAGE_SIZE)
    image_tensor = transform(image=image)['image'].unsqueeze(0).to(device)
    
    results = {}
    
    for model_name in model_names:
        print(f"\nGenerating Grad-CAM for {model_name.upper()}...")
        
        # Load model
        model = ModelFactory.create_model(
            model_name=model_name,
            num_classes=NUM_CLASSES,
            pretrained=True
        ).to(device)
        model.eval()
        
        # Generate Grad-CAM
        overlaid, heatmap, pred_class = visualize_gradcam(
            model=model,
            model_name=model_name,
            image=image,
            image_tensor=image_tensor,
            device=device
        )
        
        results[model_name] = {
            'overlaid': overlaid,
            'heatmap': heatmap,
            'prediction': pred_class
        }
        
        print(f"  Prediction: {'Cancer' if pred_class == 1 else 'No Cancer'}")
    
    # Visualize comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original CT Scan", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Each model's Grad-CAM
    for idx, (model_name, result) in enumerate(results.items()):
        row = (idx + 1) // 2
        col = (idx + 1) % 2
        
        axes[row, col].imshow(result['overlaid'])
        pred_text = 'Cancer' if result['prediction'] == 1 else 'No Cancer'
        axes[row, col].set_title(f"{model_name.upper()}\nPrediction: {pred_text}", 
                                fontsize=12, fontweight='bold')
        axes[row, col].axis('off')
    
    plt.suptitle('Grad-CAM Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = 'outputs/gradcam_demo/model_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Model comparison saved to: {save_path}")


def main():
    """Run complete Grad-CAM demo."""
    print("\n" + "="*70)
    print("GRAD-CAM EXPLAINABILITY DEMO")
    print("="*70)
    print("\nThis demo demonstrates:")
    print("  1. Creating demo CT scan images")
    print("  2. Loading a CNN model")
    print("  3. Generating Grad-CAM for single image")
    print("  4. Batch processing multiple images")
    print("  5. Creating visualizations")
    print("  6. Comparing Grad-CAM across models")
    
    # Set seed
    set_seed(42)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    try:
        # Step 1: Create demo images
        image_dir = create_demo_images()
        
        # Step 2: Load model
        model_name = 'efficientnet'
        model = load_pretrained_model(model_name, device)
        
        # Step 3: Single image Grad-CAM
        single_overlaid, single_heatmap, single_image, pred = generate_single_gradcam(
            model, model_name, f'{image_dir}/ct_scan_1.png', device
        )
        
        # Step 4: Batch generation
        results = batch_gradcam_generation(model, model_name, image_dir, device)
        
        # Step 5: Visualize
        visualize_comparison(results, model_name)
        
        # Step 6: Compare models
        compare_models_gradcam(f'{image_dir}/ct_scan_3.png', device)
        
        # Summary
        print("\n" + "="*70)
        print("DEMO COMPLETE!")
        print("="*70)
        print("\n✅ Grad-CAM demonstration successful!")
        print("\nGenerated files:")
        print("  📁 data/gradcam_demo/ - Demo CT scan images")
        print("  📊 outputs/gradcam_demo/efficientnet_gradcam_comparison.png - Batch results")
        print("  📊 outputs/gradcam_demo/model_comparison.png - Model comparison")
        print("  🖼️  outputs/gradcam_demo/*_gradcam.png - Individual overlays")
        print("\nKey Concepts Demonstrated:")
        print("  ✓ Heatmap generation showing important regions")
        print("  ✓ Overlay visualization on original images")
        print("  ✓ Batch processing multiple CT scans")
        print("  ✓ Cross-model comparison")
        print("\nClinical Interpretation:")
        print("  • Red regions = High influence on prediction")
        print("  • Blue regions = Low influence")
        print("  • Heatmap should highlight tumor locations")
        print("  • Builds trust by showing model reasoning")
        print("\nNext steps:")
        print("  1. Use real CT scans with actual tumors")
        print("  2. Validate heatmaps against radiologist annotations")
        print("  3. Integrate into demo app: python demo_app.py")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  - Install dependencies: pip install torch torchvision opencv-python matplotlib")
        print("  - Ensure model is loaded correctly")
        print("  - Check image paths exist")


if __name__ == "__main__":
    main()
