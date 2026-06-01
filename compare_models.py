"""
Model Comparison Script
=======================

Train and compare all three CNN models (EfficientNet, DenseNet, ResNet).
"""

import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.config import *
from src.models.base_models import ModelFactory
from src.models.trainer import Trainer
from src.datasets import create_dataloaders
from src.augmentations import get_train_transforms, get_val_transforms
from src.utils import set_seed, ensure_dir


def train_and_evaluate_model(model_name, train_loader, val_loader, device='cuda'):
    """Train and evaluate a single model."""
    print(f"\n{'='*70}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*70}\n")
    
    # Create model
    model = ModelFactory.create_model(
        model_name=model_name,
        num_classes=NUM_CLASSES,
        pretrained=True
    ).to(device)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        num_classes=NUM_CLASSES,
        class_weights=CLASS_WEIGHTS,
        use_amp=True
    )
    
    # Phase 1: Frozen backbone
    model.freeze_backbone()
    history_frozen = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=5,
        learning_rate=LEARNING_RATE,
        checkpoint_dir=f'checkpoints/comparison_{model_name}',
        model_name=f'{model_name}_frozen'
    )
    
    # Phase 2: Fine-tuning
    model.unfreeze_backbone()
    history_finetune = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=5,
        learning_rate=LEARNING_RATE / 10,
        checkpoint_dir=f'checkpoints/comparison_{model_name}',
        model_name=f'{model_name}_finetuned'
    )
    
    # Combine histories
    history = {
        'train_loss': history_frozen['train_loss'] + history_finetune['train_loss'],
        'val_loss': history_frozen['val_loss'] + history_finetune['val_loss'],
        'train_acc': history_frozen['train_acc'] + history_finetune['train_acc'],
        'val_acc': history_frozen['val_acc'] + history_finetune['val_acc']
    }
    
    # Get final metrics
    best_val_acc = max(history['val_acc'])
    final_val_acc = history['val_acc'][-1]
    
    return history, best_val_acc, final_val_acc


def compare_models(train_loader, val_loader, device='cuda'):
    """Train and compare all models."""
    models = ['efficientnet', 'densenet', 'resnet']
    results = {}
    
    for model_name in models:
        history, best_acc, final_acc = train_and_evaluate_model(
            model_name, train_loader, val_loader, device
        )
        results[model_name] = {
            'history': history,
            'best_val_acc': best_acc,
            'final_val_acc': final_acc
        }
    
    return results


def visualize_comparison(results):
    """Visualize model comparison."""
    ensure_dir('outputs/comparison')
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Validation Accuracy Comparison
    ax1 = axes[0, 0]
    for model_name, data in results.items():
        epochs = range(1, len(data['history']['val_acc']) + 1)
        ax1.plot(epochs, data['history']['val_acc'], label=model_name.upper(), linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Accuracy (%)')
    ax1.set_title('Validation Accuracy Comparison', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training Loss Comparison
    ax2 = axes[0, 1]
    for model_name, data in results.items():
        epochs = range(1, len(data['history']['train_loss']) + 1)
        ax2.plot(epochs, data['history']['train_loss'], label=model_name.upper(), linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Loss')
    ax2.set_title('Training Loss Comparison', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Best Accuracy Bar Chart
    ax3 = axes[1, 0]
    model_names = [m.upper() for m in results.keys()]
    best_accs = [results[m]['best_val_acc'] for m in results.keys()]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bars = ax3.bar(model_names, best_accs, color=colors, alpha=0.7)
    ax3.set_ylabel('Best Validation Accuracy (%)')
    ax3.set_title('Best Validation Accuracy by Model', fontweight='bold')
    ax3.set_ylim([0, 100])
    # Add value labels on bars
    for bar, acc in zip(bars, best_accs):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Summary Table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    table_data = []
    for model_name, data in results.items():
        table_data.append([
            model_name.upper(),
            f"{data['best_val_acc']:.2f}%",
            f"{data['final_val_acc']:.2f}%"
        ])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Model', 'Best Val Acc', 'Final Val Acc'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, len(table_data) + 1):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.suptitle('Model Comparison Summary', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = 'outputs/comparison/model_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Comparison visualization saved to: {save_path}")


def print_summary(results):
    """Print comparison summary."""
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70 + "\n")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['best_val_acc'])
    
    print("Performance Ranking:")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['best_val_acc'], reverse=True)
    
    for rank, (model_name, data) in enumerate(sorted_results, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
        print(f"{medal} {rank}. {model_name.upper():15} "
              f"Best: {data['best_val_acc']:.2f}%  "
              f"Final: {data['final_val_acc']:.2f}%")
    
    print(f"\n🏆 Best Model: {best_model[0].upper()} "
          f"(Val Accuracy: {best_model[1]['best_val_acc']:.2f}%)")


def main():
    """Run model comparison."""
    print("\n" + "="*70)
    print("CNN MODEL COMPARISON")
    print("="*70)
    print("\nComparing three models:")
    print("  • EfficientNet-B0")
    print("  • DenseNet-121")
    print("  • ResNet-50")
    print("\nThis will take approximately 10-15 minutes...\n")
    
    # Set seed
    set_seed(42)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    try:
        # Create data loaders
        print("Creating data loaders...")
        train_loader, val_loader = create_dataloaders(
            data_dir='data/demo_cnn',
            batch_size=8,
            num_workers=0,
            train_transform=get_train_transforms(IMAGE_SIZE),
            val_transform=get_val_transforms(IMAGE_SIZE)
        )
        
        # Train and compare models
        results = compare_models(train_loader, val_loader, device)
        
        # Visualize comparison
        visualize_comparison(results)
        
        # Print summary
        print_summary(results)
        
        print("\n" + "="*70)
        print("COMPARISON COMPLETE!")
        print("="*70)
        print("\n✓ All models trained and compared")
        print(f"✓ Results saved to: outputs/comparison/")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
