"""
Utility functions for the lung cancer detection project.
"""
import torch
import numpy as np
import random
import os
from pathlib import Path


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint."""
    ensure_dir(os.path.dirname(path))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path):
    """Load model checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {path} (Epoch: {epoch}, Loss: {loss:.4f})")
    return epoch, loss


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def apply_hu_windowing(image, center=-600, width=1500):
    """
    Apply HU windowing to CT scans.
    
    Args:
        image: CT image in HU units
        center: Window center
        width: Window width
    
    Returns:
        Windowed image normalized to [0, 1]
    """
    min_val = center - width // 2
    max_val = center + width // 2
    
    image = np.clip(image, min_val, max_val)
    image = (image - min_val) / (max_val - min_val)
    
    return image
