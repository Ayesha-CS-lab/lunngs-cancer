"""
Grad-CAM implementation for explainability.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM)."""
    
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model
            target_layer: Target layer for CAM (e.g., model.backbone.features[-1])
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save activations."""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        """
        Generate Class Activation Map.
        
        Args:
            input_image: Input tensor [1, C, H, W]
            target_class: Target class index (if None, use predicted class)
        
        Returns:
            cam: Class activation map
            prediction: Predicted class
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[:, target_class]
        class_score.backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy(), target_class
    
    def overlay_heatmap(self, original_image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Overlay CAM heatmap on original image.
        
        Args:
            original_image: Original image [H, W, C] in range [0, 255]
            cam: Class activation map [h, w]
            alpha: Transparency factor
            colormap: OpenCV colormap
        
        Returns:
            Overlaid image
        """
        # Resize CAM to match image size
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Ensure original image is uint8
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        else:
            original_image = original_image.astype(np.uint8)
        
        # Overlay
        overlaid = cv2.addWeighted(heatmap, alpha, original_image, 1 - alpha, 0)
        
        return overlaid, heatmap


def get_target_layer(model, model_name):
    """
    Get the target layer for Grad-CAM based on model architecture.
    
    Args:
        model: PyTorch model
        model_name: Model architecture name
    
    Returns:
        Target layer
    """
    if 'efficientnet' in model_name.lower():
        return model.backbone._conv_head
    elif 'densenet' in model_name.lower():
        return model.backbone.features[-1]
    elif 'resnet' in model_name.lower():
        return model.backbone.layer4[-1]
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")


def visualize_gradcam(
    model,
    model_name,
    image,
    image_tensor,
    target_class=None,
    save_path=None
):
    """
    Generate and visualize Grad-CAM.
    
    Args:
        model: PyTorch model
        model_name: Model architecture name
        image: Original image [H, W, C]
        image_tensor: Preprocessed image tensor [1, C, H, W]
        target_class: Target class (if None, use prediction)
        save_path: Path to save visualization
    
    Returns:
        overlaid image, heatmap, predicted class
    """
    # Get target layer
    target_layer = get_target_layer(model, model_name)
    
    # Create Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Generate CAM
    cam, pred_class = gradcam.generate_cam(image_tensor, target_class)
    
    # Overlay heatmap
    overlaid, heatmap = gradcam.overlay_heatmap(image, cam)
    
    # Visualize
    if save_path:
        from src.utils import ensure_dir
        ensure_dir(Path(save_path).parent)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image if image.max() > 1 else (image * 255).astype(np.uint8))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(heatmap)
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        axes[2].imshow(overlaid)
        axes[2].set_title(f'Overlay (Predicted: Class {pred_class})')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return overlaid, heatmap, pred_class
