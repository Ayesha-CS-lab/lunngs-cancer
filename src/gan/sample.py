"""
GAN Sample Generator
====================

Generate synthetic images using trained GAN.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gan.generator import Generator
from src.utils import ensure_dir


def load_generator(checkpoint_path, device='cuda'):
    """Load trained generator from checkpoint."""
    print(f"Loading generator from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    generator = Generator(
        latent_dim=config['latent_dim'],
        num_classes=config['num_classes'],
        img_size=config['img_size']
    ).to(device)
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    print(f"✓ Generator loaded (trained for {checkpoint['epoch']} epochs)")
    
    return generator, config


def generate_images(generator, config, num_samples, target_class, device='cuda'):
    """Generate synthetic images."""
    print(f"\nGenerating {num_samples} images for class {target_class}...")
    
    latent_dim = config['latent_dim']
    generated_images = []
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc='Generating'):
            # Sample noise
            z = torch.randn(1, latent_dim, device=device)
            label = torch.tensor([target_class], device=device)
            
            # Generate image
            gen_img = generator(z, label)
            
            # Convert to numpy
            gen_img = gen_img.squeeze(0).cpu().numpy()
            gen_img = (gen_img + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
            gen_img = np.transpose(gen_img, (1, 2, 0))
            gen_img = (gen_img * 255).astype(np.uint8)
            
            generated_images.append(gen_img)
    
    return generated_images


def save_images(images, output_dir, prefix='synthetic', target_class=1):
    """Save generated images."""
    ensure_dir(output_dir)
    
    print(f"\nSaving {len(images)} images to: {output_dir}")
    
    for i, img in enumerate(tqdm(images, desc='Saving')):
        filename = f'{prefix}_class{target_class}_{i:04d}.png'
        save_path = Path(output_dir) / filename
        cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    print(f"✓ All images saved successfully")


def main():
    """Generate synthetic images from trained GAN."""
    parser = argparse.ArgumentParser(description='Generate synthetic images using trained GAN')
    
    parser.add_argument('--checkpoint', type=str, default='checkpoints/gan/gan_best.pth',
                       help='Path to trained GAN checkpoint')
    parser.add_argument('--num_samples', type=int, default=500,
                       help='Number of samples to generate')
    parser.add_argument('--target_class', type=int, default=1,
                       help='Target class (0=no_cancer, 1=cancer)')
    parser.add_argument('--output_dir', type=str, default='data/synthetic',
                       help='Output directory for generated images')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠ CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    
    print("="*70)
    print("SYNTHETIC IMAGE GENERATION")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Target class: {args.target_class} ({'Cancer' if args.target_class == 1 else 'No Cancer'})")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {device}")
    print("="*70)
    
    # Load generator
    generator, config = load_generator(args.checkpoint, device)
    
    # Generate images
    images = generate_images(generator, config, args.num_samples, args.target_class, device)
    
    # Save images
    save_images(images, args.output_dir, target_class=args.target_class)
    
    print("\n" + "="*70)
    print("GENERATION COMPLETE!")
    print("="*70)
    print(f"✓ Generated {len(images)} synthetic images")
    print(f"✓ Saved to: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Review generated images for quality")
    print("  2. Copy to data/processed/train/cancer/ to use in training")
    print("  3. Train model: python train.py --model efficientnet")


if __name__ == '__main__':
    main()
