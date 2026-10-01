"""Unit tests for Deep Learning Architectures and Feature Extraction"""
import torch
try:
    import pytest
except ImportError:
    pytest = None
from src.models import (
    CustomBaselineCNN,
    DenseNet121Classifier,
    ResNet50Classifier,
    EfficientNetB0Classifier,
    get_model
)


def test_custom_cnn_forward_and_features():
    model = CustomBaselineCNN(num_classes=4)
    x = torch.randn(2, 3, 224, 224)

    logits = model(x)
    assert logits.shape == (2, 4)

    features = model.extract_features(x)
    assert features.shape == (2, 256)


def test_densenet_forward_and_features():
    model = DenseNet121Classifier(num_classes=4, pretrained=False)
    x = torch.randn(2, 3, 224, 224)

    logits = model(x)
    assert logits.shape == (2, 4)

    features = model.extract_features(x)
    assert features.shape == (2, 1024)


def test_resnet_forward_and_features():
    model = ResNet50Classifier(num_classes=4, pretrained=False)
    x = torch.randn(2, 3, 224, 224)

    logits = model(x)
    assert logits.shape == (2, 4)

    features = model.extract_features(x)
    assert features.shape == (2, 2048)


def test_two_phase_fine_tuning_freeze():
    model = DenseNet121Classifier(num_classes=4, pretrained=False)
    model.freeze_backbone()

    # Backbone features should have requires_grad=False
    for p in model.backbone.features.parameters():
        assert not p.requires_grad

    # Head should have requires_grad=True
    for p in model.backbone.classifier.parameters():
        assert p.requires_grad

    # Unfreeze deep block
    model.unfreeze_last_blocks(num_blocks=1)
    if hasattr(model.backbone.features, "denseblock4"):
        for p in model.backbone.features.denseblock4.parameters():
            assert p.requires_grad
