"""__init__ file for GAN module."""
from .generator import Generator
from .discriminator import Discriminator
from .train_gan import train_gan
from .sample import generate_synthetic_images

__all__ = ['Generator', 'Discriminator', 'train_gan', 'generate_synthetic_images']
