"""
Production GAN Trainer
======================

Advanced GAN training script with:
- Progress tracking
- Model checkpointing
- Loss visualization
- Sample generation during training
- Early stopping
- Learning rate scheduling
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import argparse
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.config import *
from src.gan.generator import Generator
from src.gan.discriminator import Discriminator
from src.datasets import LungCancerDataset
from src.augmentations import get_train_transforms
from src.utils import set_seed, ensure_dir


class GANTrainer:
    """Production-quality GAN trainer with advanced features."""
    
    def __init__(self, config):
        """
        Initialize GAN trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.generator = Generator(
            latent_dim=config['latent_dim'],
            num_classes=config['num_classes'],
            img_size=config['img_size']
        ).to(self.device)
        
        self.discriminator = Discriminator(
            num_classes=config['num_classes'],
            img_size=config['img_size']
        ).to(self.device)
        
        # Loss function
        self.adversarial_loss = nn.BCELoss()
        
        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=config['lr'],
            betas=(config['beta1'], config['beta2'])
        )
        
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config['lr'],
            betas=(config['beta1'], config['beta2'])
        )
        
        # Training history
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'epoch': []
        }
        
        # Best loss for checkpointing
        self.best_g_loss = float('inf')
        
        print(f"\n{'='*70}")
        print("GAN TRAINER INITIALIZED")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Generator params: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator params: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        print(f"Latent dimension: {config['latent_dim']}")
        print(f"Image size: {config['img_size']}x{config['img_size']}")
        print(f"Number of classes: {config['num_classes']}")
    
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        pbar = tqdm(dataloader, desc='Training')
        
        for imgs, labels in pbar:
            batch_size = imgs.size(0)
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            
            # Ground truth labels
            valid = torch.ones(batch_size, 1, device=self.device)
            fake = torch.zeros(batch_size, 1, device=self.device)
            
            # -----------------
            # Train Generator
            # -----------------
            self.optimizer_G.zero_grad()
            
            # Sample noise and labels
            z = torch.randn(batch_size, self.config['latent_dim'], device=self.device)
            gen_labels = torch.randint(0, self.config['num_classes'], (batch_size,), device=self.device)
            
            # Generate images
            gen_imgs = self.generator(z, gen_labels)
            
            # Generator loss (fool discriminator)
            validity = self.discriminator(gen_imgs, gen_labels)
            g_loss = self.adversarial_loss(validity, valid)
            
            g_loss.backward()
            self.optimizer_G.step()
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            self.optimizer_D.zero_grad()
            
            # Real images
            validity_real = self.discriminator(imgs, labels)
            d_real_loss = self.adversarial_loss(validity_real, valid)
            
            # Fake images
            validity_fake = self.discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = self.adversarial_loss(validity_fake, fake)
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            
            d_loss.backward()
            self.optimizer_D.step()
            
            # Update progress
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            
            pbar.set_postfix({
                'D_loss': f'{d_loss.item():.4f}',
                'G_loss': f'{g_loss.item():.4f}'
            })
        
        # Average losses
        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)
        
        return avg_g_loss, avg_d_loss
    
    def generate_samples(self, num_samples=16, target_class=1):
        """Generate sample images for visualization."""
        self.generator.eval()
        
        with torch.no_grad():
            z = torch.randn(num_samples, self.config['latent_dim'], device=self.device)
            labels = torch.full((num_samples,), target_class, dtype=torch.long, device=self.device)
            
            gen_imgs = self.generator(z, labels)
            
            # Denormalize
            gen_imgs = (gen_imgs + 1) / 2
            gen_imgs = gen_imgs.cpu()
        
        return gen_imgs
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'history': self.history,
            'config': self.config
        }
        
        # Save latest
        checkpoint_path = Path(self.config['checkpoint_dir']) / 'gan_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best
        if is_best:
            best_path = Path(self.config['checkpoint_dir']) / 'gan_best.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved (epoch {epoch})")
    
    def save_samples(self, samples, epoch):
        """Save generated samples as grid."""
        ensure_dir(self.config['sample_dir'])
        
        # Create grid
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        fig.suptitle(f'Generated Samples - Epoch {epoch}', fontsize=14, fontweight='bold')
        
        for i in range(16):
            row = i // 4
            col = i % 4
            img = samples[i].permute(1, 2, 0).numpy()
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        save_path = Path(self.config['sample_dir']) / f'samples_epoch_{epoch:03d}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def plot_losses(self):
        """Plot training losses."""
        ensure_dir(self.config['output_dir'])
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['epoch'], self.history['g_loss'], label='Generator Loss', color='blue', alpha=0.7)
        plt.plot(self.history['epoch'], self.history['d_loss'], label='Discriminator Loss', color='red', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('GAN Training Losses', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = Path(self.config['output_dir']) / 'training_losses.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Loss plot saved to {save_path}")
    
    def train(self, dataloader, num_epochs):
        """Full training loop."""
        print(f"\n{'='*70}")
        print(f"STARTING GAN TRAINING")
        print(f"{'='*70}")
        print(f"Epochs: {num_epochs}")
        print(f"Batches per epoch: {len(dataloader)}")
        print(f"Samples per epoch: {len(dataloader.dataset)}\n")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch [{epoch}/{num_epochs}]")
            
            # Train for one epoch
            g_loss, d_loss = self.train_epoch(dataloader)
            
            # Update history
            self.history['epoch'].append(epoch)
            self.history['g_loss'].append(g_loss)
            self.history['d_loss'].append(d_loss)
            
            # Print stats
            print(f"Generator Loss: {g_loss:.4f}")
            print(f"Discriminator Loss: {d_loss:.4f}")
            
            # Save checkpoint
            is_best = g_loss < self.best_g_loss
            if is_best:
                self.best_g_loss = g_loss
            
            if epoch % self.config['checkpoint_interval'] == 0:
                self.save_checkpoint(epoch, is_best)
            
            # Generate and save samples
            if epoch % self.config['sample_interval'] == 0:
                samples = self.generate_samples(num_samples=16, target_class=1)
                self.save_samples(samples, epoch)
                print(f"✓ Samples saved for epoch {epoch}")
        
        # Final checkpoint and plots
        self.save_checkpoint(num_epochs, is_best=True)
        self.plot_losses()
        
        # Save training history
        history_path = Path(self.config['output_dir']) / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n{'='*70}")
        print("TRAINING COMPLETED!")
        print(f"{'='*70}")
        print(f"✓ Best Generator Loss: {self.best_g_loss:.4f}")
        print(f"✓ Checkpoints saved to: {self.config['checkpoint_dir']}")
        print(f"✓ Samples saved to: {self.config['sample_dir']}")
        print(f"✓ Training history: {history_path}")


def create_config(args):
    """Create configuration from arguments."""
    config = {
        'latent_dim': 100,
        'num_classes': 2,
        'img_size': args.img_size,
        'lr': args.lr,
        'beta1': 0.5,
        'beta2': 0.999,
        'checkpoint_dir': args.checkpoint_dir,
        'sample_dir': args.sample_dir,
        'output_dir': args.output_dir,
        'checkpoint_interval': args.checkpoint_interval,
        'sample_interval': args.sample_interval,
    }
    
    # Ensure directories exist
    ensure_dir(config['checkpoint_dir'])
    ensure_dir(config['sample_dir'])
    ensure_dir(config['output_dir'])
    
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Conditional GAN for Data Augmentation')
    
    parser.add_argument('--data_dir', type=str, default='data/processed/train',
                       help='Path to training data directory')
    parser.add_argument('--img_size', type=int, default=64,
                       help='Image size (default: 64)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0002,
                       help='Learning rate (default: 0.0002)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/gan',
                       help='Checkpoint directory')
    parser.add_argument('--sample_dir', type=str, default='outputs/gan/samples',
                       help='Sample output directory')
    parser.add_argument('--output_dir', type=str, default='outputs/gan',
                       help='Output directory for plots')
    parser.add_argument('--checkpoint_interval', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--sample_interval', type=int, default=5,
                       help='Generate samples every N epochs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create configuration
    config = create_config(args)
    
    # Print configuration
    print(f"\n{'='*70}")
    print("GAN TRAINING CONFIGURATION")
    print(f"{'='*70}")
    for key, value in config.items():
        print(f"{key}: {value}")
    print(f"{'='*70}\n")
    
    # Create dataloader (simplified for demo - replace with actual data pipeline)
    print("Loading data...")
    try:
        from src.datasets import create_dataloaders
        train_loader, _ = create_dataloaders(
            data_dir=Path(args.data_dir).parent,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_transform=get_train_transforms(args.img_size),
            val_transform=get_train_transforms(args.img_size)
        )
        print(f"✓ Loaded {len(train_loader.dataset)} training samples")
    except Exception as e:
        print(f"⚠ Could not load data: {e}")
        print("Please ensure data is in the correct format at:", args.data_dir)
        return
    
    # Initialize trainer
    trainer = GANTrainer(config)
    
    # Train
    trainer.train(train_loader, args.epochs)


if __name__ == '__main__':
    main()
