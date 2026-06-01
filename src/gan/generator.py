"""
Generator network for Conditional GAN.
"""
import torch
import torch.nn as nn


class Generator(nn.Module):
    """Generator network for creating synthetic lung nodule images."""
    
    def __init__(self, latent_dim=100, num_classes=2, img_size=64, channels=3):
        """
        Args:
            latent_dim: Dimension of latent noise vector
            num_classes: Number of classes for conditioning
            img_size: Output image size
            channels: Number of output channels
        """
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Embed class labels
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        
        # Initial size
        self.init_size = img_size // 4
        self.fc = nn.Linear(latent_dim * 2, 128 * self.init_size ** 2)
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        """
        Forward pass.
        
        Args:
            noise: Random noise tensor [B, latent_dim]
            labels: Class labels [B]
        
        Returns:
            Generated images [B, C, H, W]
        """
        # Embed labels and concatenate with noise
        label_embedding = self.label_embedding(labels)
        gen_input = torch.cat([noise, label_embedding], dim=1)
        
        # Generate image
        out = self.fc(gen_input)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        
        return img
