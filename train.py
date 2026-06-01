"""
Main training script for lung cancer detection.
"""
import argparse
import torch
from pathlib import Path

from configs.config import *
from src.utils import set_seed, ensure_dir
from src.datasets import create_dataloaders
from src.augmentations import get_train_transforms, get_val_transforms
from src.models import get_model, Trainer
from src.evaluation import evaluate_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train lung cancer detection model')
    
    parser.add_argument('--model', type=str, default='efficientnet',
                        choices=['efficientnet', 'densenet', 'resnet'],
                        help='Model architecture')
    parser.add_argument('--data-dir', type=str, default=PROCESSED_DATA_DIR,
                        help='Path to processed data')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Random seed')
    parser.add_argument('--checkpoint-dir', type=str, default=CHECKPOINT_DIR,
                        help='Checkpoint directory')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                        help='Output directory for plots')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create directories
    ensure_dir(args.checkpoint_dir)
    ensure_dir(args.output_dir)
    
    print(f"\n{'='*70}")
    print(f"Training {args.model.upper()} Model")
    print(f"{'='*70}\n")
    
    print(f"Device: {DEVICE}")
    print(f"Data directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}\n")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        train_transform=get_train_transforms(IMAGE_SIZE),
        val_transform=get_val_transforms(IMAGE_SIZE)
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}\n")
    
    # Create model
    print(f"Creating {args.model} model...")
    model = get_model(args.model, num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True)
    
    from src.utils import count_parameters
    print(f"Trainable parameters: {count_parameters(model):,}\n")
    
    # Create trainer
    trainer = Trainer(
        model,
        device=DEVICE,
        learning_rate=args.lr,
        weight_decay=WEIGHT_DECAY,
        class_weights=CLASS_WEIGHTS,
        use_amp=True
    )
    
    # Train with frozen backbone
    print("\n" + "="*70)
    print("PHASE 1: Training with Frozen Backbone")
    print("="*70 + "\n")
    
    best_model_path = trainer.fit(
        train_loader,
        val_loader,
        num_epochs=args.epochs // 2,
        save_dir=args.checkpoint_dir,
        model_name=f"{args.model}_frozen",
        patience=PATIENCE
    )
    
    # Fine-tune with unfrozen backbone
    print("\n" + "="*70)
    print("PHASE 2: Fine-tuning with Unfrozen Backbone")
    print("="*70 + "\n")
    
    model.unfreeze_backbone()
    
    # Reduce learning rate for fine-tuning
    for param_group in trainer.optimizer.param_groups:
        param_group['lr'] = args.lr / 10
    
    print(f"Learning rate reduced to {args.lr / 10}\n")
    
    best_model_path = trainer.fit(
        train_loader,
        val_loader,
        num_epochs=args.epochs // 2,
        save_dir=args.checkpoint_dir,
        model_name=f"{args.model}_finetuned",
        patience=PATIENCE
    )
    
    # Evaluate
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70 + "\n")
    
    # Load best model
    from src.utils import load_checkpoint
    load_checkpoint(model, None, best_model_path)
    
    # Evaluate on validation set
    metrics = evaluate_model(
        model,
        val_loader,
        device=DEVICE,
        output_dir=f"{args.output_dir}/{args.model}"
    )
    
    print(f"\n✓ Training completed!")
    print(f"✓ Best model saved to: {best_model_path}")
    print(f"✓ Evaluation plots saved to: {args.output_dir}/{args.model}")


if __name__ == "__main__":
    main()
