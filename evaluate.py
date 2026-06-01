"""
Evaluation script for trained models.
"""
import argparse
import torch
from pathlib import Path

from configs.config import *
from src.utils import set_seed, load_checkpoint
from src.datasets import create_dataloaders
from src.augmentations import get_val_transforms
from src.models import get_model
from src.evaluation import evaluate_model, ModelEvaluator
from src.models.inference import predict


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    
    parser.add_argument('--model', type=str, required=True,
                        choices=['efficientnet', 'densenet', 'resnet'],
                        help='Model architecture')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default=PROCESSED_DATA_DIR,
                        help='Path to processed data')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                        help='Output directory for plots')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Random seed')
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    print(f"\n{'='*70}")
    print(f"Evaluating {args.model.upper()} Model")
    print(f"{'='*70}\n")
    
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {DEVICE}\n")
    
    # Create validation data loader
    print("Creating data loader...")
    _, val_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        val_transform=get_val_transforms(IMAGE_SIZE)
    )
    
    print(f"Validation batches: {len(val_loader)}\n")
    
    # Load model
    print(f"Loading {args.model} model...")
    model = get_model(args.model, num_classes=NUM_CLASSES, pretrained=False)
    load_checkpoint(model, None, args.checkpoint)
    
    # Evaluate
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70 + "\n")
    
    metrics = evaluate_model(
        model,
        val_loader,
        device=DEVICE,
        output_dir=f"{args.output_dir}/evaluation"
    )
    
    print(f"\n✓ Evaluation completed!")
    print(f"✓ Results saved to: {args.output_dir}/evaluation")


if __name__ == "__main__":
    main()
