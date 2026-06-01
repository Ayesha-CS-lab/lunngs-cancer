"""
CNN Model Training Demo
=======================

This script demonstrates training a single CNN model (EfficientNet, DenseNet, or ResNet)
with all the production features:
- Two-phase training (frozen → fine-tuned)
- Mixed precision (AMP)
- Class imbalance handling
- Learning rate scheduling
- Early stopping
- Comprehensive logging
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.config import *
from src.models.base_models import ModelFactory
from src.models.trainer import Trainer
from src.datasets import create_dataloaders
from src.augmentations import get_train_transforms, get_val_transforms
from src.utils import set_seed, ensure_dir


def create_dummy_dataset():
    """Create dummy dataset for demonstration."""
    print("\n" + "="*70)
    print("STEP 1: Creating Dummy Dataset")
    print("="*70 + "\n")
    
    ensure_dir('data/demo_cnn/train/no_cancer')
    ensure_dir('data/demo_cnn/train/cancer')
    ensure_dir('data/demo_cnn/val/no_cancer')
    ensure_dir('data/demo_cnn/val/cancer')
    
    print("Creating dummy CT scan images...")
    
    # Training images
    for i in range(30):
        # No cancer (more uniform intensity)
        img = np.random.randint(80, 120, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(f'data/demo_cnn/train/no_cancer/img_{i:03d}.png', img)
        
        # Cancer (with bright spots simulating tumors)
        img = np.random.randint(100, 150, (224, 224, 3), dtype=np.uint8)
        center = (np.random.randint(80, 140), np.random.randint(80, 140))
        radius = np.random.randint(10, 20)
        cv2.circle(img, center, radius, (255, 255, 255), -1)
        cv2.imwrite(f'data/demo_cnn/train/cancer/img_{i:03d}.png', img)
    
    # Validation images
    for i in range(10):
        img = np.random.randint(80, 120, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(f'data/demo_cnn/val/no_cancer/img_{i:03d}.png', img)
        
        img = np.random.randint(100, 150, (224, 224, 3), dtype=np.uint8)
        center = (np.random.randint(80, 140), np.random.randint(80, 140))
        radius = np.random.randint(10, 20)
        cv2.circle(img, center, radius, (255, 255, 255), -1)
        cv2.imwrite(f'data/demo_cnn/val/cancer/img_{i:03d}.png', img)
    
    print("✓ Created 80 dummy images")
    print("  Training: 30 no_cancer + 30 cancer = 60 images")
    print("  Validation: 10 no_cancer + 10 cancer = 20 images")
    
    return 'data/demo_cnn'


def create_data_loaders(data_dir, batch_size=8):
    """Create train and validation data loaders."""
    print("\n" + "="*70)
    print("STEP 2: Creating DataLoaders")
    print("="*70 + "\n")
    
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=0,  # 0 for demo to avoid issues
        train_transform=get_train_transforms(IMAGE_SIZE),
        val_transform=get_val_transforms(IMAGE_SIZE)
    )
    
    print(f"✓ DataLoaders created")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    
    return train_loader, val_loader


def initialize_model(model_name='efficientnet', device='cuda'):
    """Initialize CNN model."""
    print("\n" + "="*70)
    print(f"STEP 3: Initializing {model_name.upper()} Model")
    print("="*70 + "\n")
    
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("⚠ CUDA not available, using CPU")
    
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Pretrained: True\n")
    
    # Create model
    model = ModelFactory.create_model(
        model_name=model_name,
        num_classes=NUM_CLASSES,
        pretrained=True
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def train_model_demo(model, train_loader, val_loader, model_name='efficientnet', device='cuda'):
    """Train model with two-phase approach."""
    print("\n" + "="*70)
    print("STEP 4: Training Model (Two-Phase)")
    print("="*70 + "\n")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        num_classes=NUM_CLASSES,
        class_weights=CLASS_WEIGHTS,
        use_amp=True  # Mixed precision
    )
    
    # Phase 1: Frozen backbone
    print("\n" + "-"*70)
    print("PHASE 1: Training with Frozen Backbone (Transfer Learning)")
    print("-"*70 + "\n")
    
    model.freeze_backbone()
    
    frozen_epochs = 5  # Reduced for demo
    history_frozen = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=frozen_epochs,
        learning_rate=LEARNING_RATE,
        checkpoint_dir=f'checkpoints/demo_{model_name}',
        model_name=f'{model_name}_frozen'
    )
    
    # Phase 2: Fine-tuning
    print("\n" + "-"*70)
    print("PHASE 2: Fine-tuning with Unfrozen Backbone")
    print("-"*70 + "\n")
    
    model.unfreeze_backbone()
    
    finetune_epochs = 5  # Reduced for demo
    finetune_lr = LEARNING_RATE / 10
    
    print(f"Reduced learning rate: {finetune_lr}")
    
    history_finetune = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=finetune_epochs,
        learning_rate=finetune_lr,
        checkpoint_dir=f'checkpoints/demo_{model_name}',
        model_name=f'{model_name}_finetuned'
    )
    
    # Combine histories
    history = {
        'train_loss': history_frozen['train_loss'] + history_finetune['train_loss'],
        'val_loss': history_frozen['val_loss'] + history_finetune['val_loss'],
        'train_acc': history_frozen['train_acc'] + history_finetune['train_acc'],
        'val_acc': history_frozen['val_acc'] + history_finetune['val_acc']
    }
    
    print("\n✓ Training completed!")
    
    return history


def visualize_results(history, model_name='efficientnet'):
    """Visualize training results."""
    print("\n" + "="*70)
    print("STEP 5: Visualizing Training Results")
    print("="*70 + "\n")
    
    ensure_dir('outputs/demo_cnn')
    
    epochs = range(1, len(history['train_loss']) + 1)
    phase_split = len(history['train_loss']) // 2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.axvline(x=phase_split, color='gray', linestyle='--', alpha=0.5)
    ax1.text(phase_split/2, max(history['train_loss'])*0.95, 'Phase 1\n(Frozen)', 
             ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax1.text(phase_split + (len(epochs)-phase_split)/2, max(history['train_loss'])*0.95, 'Phase 2\n(Fine-tuned)', 
             ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{model_name.upper()} - Training Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    ax2.axvline(x=phase_split, color='gray', linestyle='--', alpha=0.5)
    ax2.text(phase_split/2, min(history['train_acc'])*1.02, 'Phase 1\n(Frozen)', 
             ha='center', va='bottom', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.text(phase_split + (len(epochs)-phase_split)/2, min(history['train_acc'])*1.02, 'Phase 2\n(Fine-tuned)', 
             ha='center', va='bottom', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title(f'{model_name.upper()} - Training Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f'outputs/demo_cnn/{model_name}_training_results.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved to: {save_path}")
    
    # Print summary
    print("\nTraining Summary:")
    print(f"  Best Train Accuracy: {max(history['train_acc']):.2f}%")
    print(f"  Best Val Accuracy: {max(history['val_acc']):.2f}%")
    print(f"  Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final Val Loss: {history['val_loss'][-1]:.4f}")


def test_model(model, val_loader, device='cuda'):
    """Test model on validation set."""
    print("\n" + "="*70)
    print("STEP 6: Testing Model")
    print("="*70 + "\n")
    
    model.eval()
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    
    print(f"✓ Test Accuracy: {accuracy:.2f}%")
    print(f"  Correct predictions: {correct}/{total}")
    
    # Confusion matrix (simple version)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\nConfusion Matrix:")
    print(f"               Predicted")
    print(f"              No Cancer  Cancer")
    print(f"Actual No Cancer:  {cm[0,0]:3d}      {cm[0,1]:3d}")
    print(f"       Cancer:      {cm[1,0]:3d}      {cm[1,1]:3d}")
    
    return accuracy


def main():
    """Run complete CNN training demo."""
    print("\n" + "="*70)
    print("CNN MODEL TRAINING DEMO")
    print("="*70)
    print("\nThis demo will:")
    print("  1. Create dummy CT scan dataset")
    print("  2. Set up DataLoaders")
    print("  3. Initialize CNN model (EfficientNet)")
    print("  4. Train with two-phase approach")
    print("  5. Visualize training curves")
    print("  6. Test final model")
    
    # Set seed
    set_seed(42)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Step 1: Create dataset
        data_dir = create_dummy_dataset()
        
        # Step 2: Create data loaders
        train_loader, val_loader = create_data_loaders(data_dir, batch_size=8)
        
        # Step 3: Initialize model
        model_name = 'efficientnet'
        model = initialize_model(model_name, device)
        
        # Step 4: Train model
        history = train_model_demo(model, train_loader, val_loader, model_name, device)
        
        # Step 5: Visualize results
        visualize_results(history, model_name)
        
        # Step 6: Test model
        test_accuracy = test_model(model, val_loader, device)
        
        # Summary
        print("\n" + "="*70)
        print("DEMO COMPLETE!")
        print("="*70)
        print("\n✅ CNN training demonstration successful!")
        print("\nGenerated files:")
        print("  📁 data/demo_cnn/ - Dummy dataset")
        print("  💾 checkpoints/demo_efficientnet/ - Model checkpoints")
        print("  📊 outputs/demo_cnn/efficientnet_training_results.png - Training curves")
        print("\nNext steps:")
        print("  1. Replace dummy data with real CT scans")
        print("  2. Train for full 50 epochs: python train.py --model efficientnet")
        print("  3. Try other models: --model densenet or --model resnet")
        print("  4. Evaluate: python evaluate.py --model efficientnet")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  - Install dependencies: pip install torch torchvision efficientnet-pytorch")
        print("  - Check CUDA availability if using GPU")
        print("  - Verify src/models modules are available")


if __name__ == "__main__":
    main()
