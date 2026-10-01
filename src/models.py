"""
Deep Learning Architectures for Chest X-Ray Classification

Includes:
- Baseline Custom CNN (4 Conv blocks + BatchNorm + Dropout)
- DenseNet-121 (Standard benchmark in Chest Radiography / CheXNet)
- ResNet-50 (Residual skip connections)
- EfficientNet-B0 (Compound scaling)
- Two-phase fine-tuning support (Freeze backbone -> Fine-tune)
- Dense feature embedding extraction for Multimodal ML fusion
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple, Optional


class BaseMedicalCNN(nn.Module):
    """Base class for chest X-ray classifiers with feature extraction & freeze controls."""
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def freeze_backbone(self):
        raise NotImplementedError

    def unfreeze_last_blocks(self, num_blocks: int = 1):
        raise NotImplementedError

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True


class CustomBaselineCNN(BaseMedicalCNN):
    """
    4-Block Convolutional Neural Network baseline.
    Demonstrates fundamental inductive biases: Conv2d -> BatchNorm -> LeakyReLU -> MaxPool -> Dropout.
    """
    def __init__(self, num_classes: int = 4, in_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes

        # Block 1: 224x224 -> 112x112
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Block 2: 112x112 -> 56x56
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Block 3: 56x56 -> 28x28
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Block 4: 28x28 -> 14x14
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 256

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        return self.classifier[1:](feat)

    def freeze_backbone(self):
        for block in [self.block1, self.block2, self.block3, self.block4]:
            for p in block.parameters():
                p.requires_grad = False

    def unfreeze_last_blocks(self, num_blocks: int = 1):
        if num_blocks >= 1:
            for p in self.block4.parameters():
                p.requires_grad = True
        if num_blocks >= 2:
            for p in self.block3.parameters():
                p.requires_grad = True


class DenseNet121Classifier(BaseMedicalCNN):
    """
    DenseNet-121 Architecture.
    Direct connections from each layer to all subsequent layers inside dense blocks.
    CheXNet-proven architecture for thoracic radiographs.
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes

        try:
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            self.backbone = models.densenet121(weights=weights)
        except Exception:
            self.backbone = models.densenet121(weights="IMAGENET1K_V1" if pretrained else None)

        self.feature_dim = self.backbone.classifier.in_features  # 1024

        # Replace classification head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.features(x)
        out = nn.functional.relu(features, inplace=False)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        return torch.flatten(out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def unfreeze_last_blocks(self, num_blocks: int = 1):
        """Unfreezes denseblock4 and norm5."""
        if hasattr(self.backbone.features, "denseblock4"):
            for param in self.backbone.features.denseblock4.parameters():
                param.requires_grad = True
        if hasattr(self.backbone.features, "norm5"):
            for param in self.backbone.features.norm5.parameters():
                param.requires_grad = True


class ResNet50Classifier(BaseMedicalCNN):
    """
    ResNet-50 Architecture.
    Uses residual identity shortcut connections: F(x) + x.
    Provides gradient highways, preventing vanishing gradients in deep layers.
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes

        try:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
        except Exception:
            self.backbone = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)

        self.feature_dim = self.backbone.fc.in_features  # 2048

        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False
            else:
                param.requires_grad = True

    def unfreeze_last_blocks(self, num_blocks: int = 1):
        """Unfreezes layer4."""
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True


class EfficientNetB0Classifier(BaseMedicalCNN):
    """
    EfficientNet-B0 Architecture.
    Compound scaling balancing network depth, width, and image resolution.
    Highest accuracy per parameter for edge / clinical deployment.
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes

        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
        except Exception:
            self.backbone = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)

        self.feature_dim = self.backbone.classifier[1].in_features  # 1280

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.features(x)
        pooled = self.backbone.avgpool(features)
        return torch.flatten(pooled, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def unfreeze_last_blocks(self, num_blocks: int = 1):
        """Unfreezes top feature layers (stages 7 and 8)."""
        for param in self.backbone.features[-2:].parameters():
            param.requires_grad = True


def get_model(
    model_name: str,
    num_classes: int = 4,
    pretrained: bool = True
) -> BaseMedicalCNN:
    """
    Model factory supporting:
    - 'densenet121'
    - 'resnet50'
    - 'efficientnet_b0'
    - 'custom_cnn'
    """
    name = model_name.lower().strip()
    if "dense" in name:
        return DenseNet121Classifier(num_classes=num_classes, pretrained=pretrained)
    elif "res" in name:
        return ResNet50Classifier(num_classes=num_classes, pretrained=pretrained)
    elif "effic" in name:
        return EfficientNetB0Classifier(num_classes=num_classes, pretrained=pretrained)
    elif "custom" in name or "base" in name:
        return CustomBaselineCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose from densenet121, resnet50, efficientnet_b0, custom_cnn")
