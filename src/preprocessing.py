"""
Data preprocessing and loading module.
"""
import cv2
import numpy as np
from pathlib import Path
import pydicom
from PIL import Image


def load_image(image_path, image_size=224):
    """
    Load and preprocess an image (supports PNG, JPEG, DICOM).
    
    Args:
        image_path: Path to image file
        image_size: Target size for resizing
    
    Returns:
        Preprocessed image as numpy array
    """
    image_path = Path(image_path)
    
    if image_path.suffix.lower() == '.dcm':
        # Load DICOM
        dcm = pydicom.dcmread(str(image_path))
        image = dcm.pixel_array.astype(np.float32)
        
        # Convert to HU if possible
        if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
            image = image * dcm.RescaleSlope + dcm.RescaleIntercept
            
        # Apply HU windowing (lung window)
        from src.utils import apply_hu_windowing
        image = apply_hu_windowing(image)
        
        # Convert to 3 channels
        image = np.stack([image, image, image], axis=-1)
    else:
        # Load PNG/JPEG
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
    
    # Resize
    image = cv2.resize(image, (image_size, image_size))
    
    return image


def normalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Normalize image using ImageNet statistics.
    
    Args:
        image: Image array [H, W, C]
        mean: Mean values for normalization
        std: Standard deviation values
    
    Returns:
        Normalized image
    """
    image = (image - np.array(mean)) / np.array(std)
    return image


def save_processed_image(image, save_path):
    """Save processed image."""
    from src.utils import ensure_dir
    ensure_dir(Path(save_path).parent)
    
    # Denormalize if needed
    if image.min() < 0:
        image = (image - image.min()) / (image.max() - image.min())
    
    image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save(save_path)
