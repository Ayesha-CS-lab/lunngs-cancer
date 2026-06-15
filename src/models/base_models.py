"""
Base deep learning models with transfer learning.
"""
import torch
import torch.nn as nn
import torchvision.models as models
import timm


class EfficientNetModel(nn.Module):
    """EfficientNet-based classifier."""

    def __init__(self, model_name='efficientnet_b0', num_classes=2, pretrained=True):
        super(EfficientNetModel, self).__init__()
        # timm is pre-installed on Kaggle; num_classes=0 removes the built-in head
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        num_ftrs = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))

    def freeze_backbone(self):
        """Freeze backbone weights for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        """Unfreeze all weights for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True


class DenseNetModel(nn.Module):
    """DenseNet-based classifier."""
    
    def __init__(self, model_name='densenet121', num_classes=2, pretrained=True):
        super(DenseNetModel, self).__init__()
        
        # Load pretrained model
        if model_name == 'densenet121':
            self.backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_name == 'densenet169':
            self.backbone = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Get number of features
        num_ftrs = self.backbone.classifier.in_features
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze backbone weights."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
    
    def unfreeze_backbone(self):
        """Unfreeze all weights."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class ResNetModel(nn.Module):
    """ResNet-based classifier."""
    
    def __init__(self, model_name='resnet50', num_classes=2, pretrained=True):
        super(ResNetModel, self).__init__()
        
        # Load pretrained model
        if model_name == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Get number of features
        num_ftrs = self.backbone.fc.in_features
        
        # Replace classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze backbone weights."""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all weights."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class ModelFactory:
    """Factory for creating models."""
    
    @staticmethod
    def create_model(model_type, num_classes=2, pretrained=True):
        """
        Create a model by type.
        
        Args:
            model_type: 'efficientnet', 'densenet', or 'resnet'
            num_classes: Number of output classes
            pretrained: Use pretrained weights
        
        Returns:
            Model instance
        """
        if model_type == 'efficientnet':
            return EfficientNetModel(num_classes=num_classes, pretrained=pretrained)
        elif model_type == 'densenet':
            return DenseNetModel(num_classes=num_classes, pretrained=pretrained)
        elif model_type == 'resnet':
            return ResNetModel(num_classes=num_classes, pretrained=pretrained)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def get_model(model_name, num_classes=2, pretrained=True, freeze_backbone=True):
    """
    Get a model ready for training.
    
    Args:
        model_name: Model architecture name
        num_classes: Number of classes
        pretrained: Use pretrained weights
        freeze_backbone: Freeze backbone for transfer learning
    
    Returns:
        Model instance
    """
    model = ModelFactory.create_model(model_name, num_classes, pretrained)
    
    if freeze_backbone and pretrained:
        model.freeze_backbone()
        print(f"Created {model_name} with frozen backbone")
    else:
        print(f"Created {model_name} with unfrozen backbone")
    
    return model


if __name__ == "__main__":
    # Test models
    from src.utils import count_parameters
    
    for model_type in ['efficientnet', 'densenet', 'resnet']:
        model = get_model(model_type)
        print(f"{model_type}: {count_parameters(model):,} trainable parameters")
