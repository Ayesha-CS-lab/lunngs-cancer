"""
Discriminator network for Conditional GAN.
"""
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """Discriminator network for GAN."""
    
    def __init__(self, num_classes=2, img_size=64, channels=3):
        """
        Args:
            num_classes: Number of classes for conditioning
            img_size: Input image size
            channels: Number of input channels
        """
        super(Discriminator, self).__init__()
        
        self.num_classes = num_classes
        
        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, img_size * img_size)
        
        # Convolutional layers
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            block.append(nn.Dropout2d(0.25))
            return block
        
        self.conv_blocks = nn.Sequential(
            *discriminator_block(channels + 1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        
        # Calculate output shape
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, labels):
        """
        Forward pass.
        
        Args:
            img: Input images [B, C, H, W]
            labels: Class labels [B]
        
        Returns:
            Probability of being real [B, 1]
        """
        # Embed labels and reshape
        label_embedding = self.label_embedding(labels)
        label_embedding = label_embedding.view(labels.size(0), 1, img.size(2), img.size(3))
        
        # Concatenate image and label
        d_in = torch.cat([img, label_embedding], dim=1)
        
        # Discriminate
        out = self.conv_blocks(d_in)
        out = out.view(out.size(0), -1)
        validity = self.adv_layer(out)
        
        return validity
