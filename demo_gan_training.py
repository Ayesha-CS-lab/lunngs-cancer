"""
GAN Training Demo Script
========================

This script demonstrates how to train the Conditional GAN for data augmentation.
It includes:
1. Setting up training data
2. Training the GAN
3. Generating synthetic images
4. Visualizing results
"""

import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.config import *
from src.gan.generator import Generator
from src.gan.discriminator import Discriminator
from src.utils import set_seed, ensure_dir


def create_dummy_training_data():
    """Create dummy data for GAN training demonstration."""
    print("\n" + "="*70)
    print("STEP 1: Creating Dummy Training Data")
    print("="*70 + "\n")
    
    ensure_dir('data/demo_gan/cancer')
    ensure_dir('data/demo_gan/no_cancer')
    
    print("Creating 20 dummy cancer images and 20 no-cancer images...")
    
    # Create cancer images (bright spots to simulate tumors)
    for i in range(20):
        img = np.random.randint(100, 150, (64, 64, 3), dtype=np.uint8)
        # Add bright spot (simulating tumor)
        center = (np.random.randint(20, 44), np.random.randint(20, 44))
        radius = np.random.randint(5, 10)
        cv2.circle(img, center, radius, (255, 255, 255), -1)
        cv2.imwrite(f'data/demo_gan/cancer/cancer_{i:03d}.png', img)
    
    # Create no-cancer images (more uniform)
    for i in range(20):
        img = np.random.randint(80, 120, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(f'data/demo_gan/no_cancer/no_cancer_{i:03d}.png', img)
    
    print("✓ Created 40 training images")
    print("  - 20 cancer images (with bright spots)")
    print("  - 20 no-cancer images (uniform)")
    return 'data/demo_gan'


def create_simple_dataloader(data_dir, batch_size=4):
    """Create a simple dataloader for GAN training."""
    print("\n" + "="*70)
    print("STEP 2: Creating DataLoader")
    print("="*70 + "\n")
    
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    
    class SimpleImageDataset(Dataset):
        def __init__(self, data_dir):
            self.image_paths = []
            self.labels = []
            
            # Load cancer images (label=1)
            cancer_dir = Path(data_dir) / 'cancer'
            for img_path in cancer_dir.glob('*.png'):
                self.image_paths.append(str(img_path))
                self.labels.append(1)
            
            # Load no-cancer images (label=0)
            no_cancer_dir = Path(data_dir) / 'no_cancer'
            for img_path in no_cancer_dir.glob('*.png'):
                self.image_paths.append(str(img_path))
                self.labels.append(0)
            
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            img = cv2.imread(self.image_paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(img)
            label = self.labels[idx]
            return img, label
    
    dataset = SimpleImageDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"✓ DataLoader created")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Batches per epoch: {len(dataloader)}")
    
    return dataloader


def train_gan_demo(dataloader, num_epochs=50, device='cuda'):
    """Train GAN for demonstration."""
    print("\n" + "="*70)
    print("STEP 3: Training Conditional GAN")
    print("="*70 + "\n")
    
    # Check device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("⚠ CUDA not available, using CPU")
    
    print(f"Device: {device}")
    print(f"Training for {num_epochs} epochs\n")
    
    # Initialize models
    latent_dim = 100
    num_classes = 2
    img_size = 64
    
    generator = Generator(latent_dim, num_classes, img_size).to(device)
    discriminator = Discriminator(num_classes, img_size).to(device)
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}\n")
    
    # Loss and optimizers
    adversarial_loss = torch.nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Training history
    g_losses = []
    d_losses = []
    
    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.size(0)
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            # Ground truths
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)
            
            # -----------------
            # Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Sample noise and labels
            z = torch.randn(batch_size, latent_dim, device=device)
            gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)
            
            # Generate images
            gen_imgs = generator(z, gen_labels)
            
            # Generator loss
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)
            
            g_loss.backward()
            optimizer_G.step()
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Real images
            validity_real = discriminator(imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)
            
            # Fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            
            d_loss.backward()
            optimizer_D.step()
            
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
        
        # Average losses
        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)
        
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}")
    
    print("\n✓ Training completed!")
    
    # Save models
    ensure_dir('checkpoints/demo_gan')
    torch.save(generator.state_dict(), 'checkpoints/demo_gan/generator.pth')
    torch.save(discriminator.state_dict(), 'checkpoints/demo_gan/discriminator.pth')
    print("✓ Models saved to checkpoints/demo_gan/")
    
    return generator, discriminator, g_losses, d_losses


def generate_synthetic_images(generator, num_samples=10, target_class=1, device='cuda'):
    """Generate synthetic images using trained generator."""
    print("\n" + "="*70)
    print("STEP 4: Generating Synthetic Images")
    print("="*70 + "\n")
    
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    generator.eval()
    latent_dim = 100
    
    print(f"Generating {num_samples} synthetic images for class {target_class}")
    print(f"Class {target_class} = {'Cancer' if target_class == 1 else 'No Cancer'}\n")
    
    synthetic_images = []
    
    with torch.no_grad():
        for i in range(num_samples):
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
            
            synthetic_images.append(gen_img)
    
    print(f"✓ Generated {len(synthetic_images)} synthetic images")
    
    # Save synthetic images
    ensure_dir('outputs/demo_gan/synthetic')
    for i, img in enumerate(synthetic_images):
        cv2.imwrite(f'outputs/demo_gan/synthetic/synthetic_{target_class}_{i:03d}.png', 
                   cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    print(f"✓ Saved to outputs/demo_gan/synthetic/")
    
    return synthetic_images


def visualize_results(synthetic_images, g_losses, d_losses):
    """Visualize GAN results."""
    print("\n" + "="*70)
    print("STEP 5: Visualizing Results")
    print("="*70 + "\n")
    
    ensure_dir('outputs/demo_gan')
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Training losses
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(g_losses, label='Generator Loss', color='blue', alpha=0.7)
    ax1.plot(d_losses, label='Discriminator Loss', color='red', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Losses Over Time', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2-4: Sample generated images
    for i in range(min(3, len(synthetic_images))):
        ax = plt.subplot(2, 2, i + 2)
        ax.imshow(synthetic_images[i])
        ax.set_title(f'Generated Sample {i+1}', fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/demo_gan/gan_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Visualization saved to outputs/demo_gan/gan_results.png")
    
    # Create comparison grid
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Generated Synthetic Images (Cancer Class)', fontsize=14, fontweight='bold')
    
    for i in range(min(10, len(synthetic_images))):
        row = i // 5
        col = i % 5
        axes[row, col].imshow(synthetic_images[i])
        axes[row, col].axis('off')
        axes[row, col].set_title(f'Sample {i+1}')
    
    plt.tight_layout()
    plt.savefig('outputs/demo_gan/generated_grid.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Grid saved to outputs/demo_gan/generated_grid.png")


def main():
    """Run complete GAN training demo."""
    print("\n" + "="*70)
    print("CONDITIONAL GAN TRAINING DEMO")
    print("="*70)
    print("\nThis demo will:")
    print("  1. Create dummy training data")
    print("  2. Set up DataLoader")
    print("  3. Train Conditional GAN")
    print("  4. Generate synthetic images")
    print("  5. Visualize results")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Step 1: Create training data
        data_dir = create_dummy_training_data()
        
        # Step 2: Create dataloader
        dataloader = create_simple_dataloader(data_dir, batch_size=4)
        
        # Step 3: Train GAN
        generator, discriminator, g_losses, d_losses = train_gan_demo(
            dataloader, 
            num_epochs=50,  # Reduced for demo
            device=device
        )
        
        # Step 4: Generate synthetic images
        synthetic_images = generate_synthetic_images(
            generator, 
            num_samples=10, 
            target_class=1,  # Cancer class
            device=device
        )
        
        # Step 5: Visualize results
        visualize_results(synthetic_images, g_losses, d_losses)
        
        # Summary
        print("\n" + "="*70)
        print("DEMO COMPLETE!")
        print("="*70)
        print("\n✅ GAN training demonstration successful!")
        print("\nGenerated files:")
        print("  📁 data/demo_gan/ - Training data")
        print("  💾 checkpoints/demo_gan/ - Trained models")
        print("  🖼️  outputs/demo_gan/synthetic/ - Generated images")
        print("  📊 outputs/demo_gan/gan_results.png - Results visualization")
        print("  📊 outputs/demo_gan/generated_grid.png - Generated samples grid")
        print("\nNext steps:")
        print("  1. Use real cancer data for better results")
        print("  2. Train for more epochs (100-200)")
        print("  3. Generate synthetic images to balance dataset")
        print("  4. Use in training: python train.py --model efficientnet")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  - Ensure PyTorch is installed: pip install torch torchvision")
        print("  - Check CUDA availability if using GPU")
        print("  - Verify all dependencies: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
