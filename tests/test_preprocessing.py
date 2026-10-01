"""Unit tests for Chest X-Ray Preprocessing Pipeline"""
import numpy as np
import torch
try:
    import pytest
except ImportError:
    pytest = None
from src.preprocessing import (
    apply_clahe,
    ensure_3_channels,
    preprocess_xray_image,
    denormalize_image,
    validate_chest_xray,
    MedicalAugmentationPipeline,
    IMAGENET_MEAN,
    IMAGENET_STD
)


def test_ensure_3_channels_grayscale():
    gray = np.ones((100, 100), dtype=np.uint8) * 128
    three_ch = ensure_3_channels(gray)
    assert three_ch.shape == (100, 100, 3)
    assert np.all(three_ch[:, :, 0] == 128)
    assert np.all(three_ch[:, :, 1] == 128)
    assert np.all(three_ch[:, :, 2] == 128)


def test_apply_clahe():
    img = np.random.randint(50, 200, (120, 120), dtype=np.uint8)
    enhanced = apply_clahe(img, clip_limit=2.0)
    assert enhanced.shape == (120, 120)
    assert enhanced.dtype == np.uint8


def test_preprocess_xray_image():
    dummy = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    tensor, orig, clahe_img = preprocess_xray_image(dummy, target_size=(224, 224), use_clahe=True)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)
    assert orig.shape == (256, 256, 3)
    assert clahe_img.shape == (256, 256, 3)


def test_denormalize_image():
    dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    tensor, _, _ = preprocess_xray_image(dummy, target_size=(224, 224))
    restored = denormalize_image(tensor)
    assert restored.shape == (224, 224, 3)
    assert restored.dtype == np.uint8


def test_validate_chest_xray_filters():
    # 1. Color photo / selfie test (should be rejected)
    color_photo = np.zeros((300, 300, 3), dtype=np.uint8)
    color_photo[:, :] = [120, 180, 230] # High saturation color
    valid_color, reason_c, _ = validate_chest_xray(color_photo)
    assert not valid_color
    assert "color image" in reason_c.lower() or "chrominance" in reason_c.lower()

    # 2. Blank uniform document test (should be rejected)
    blank_doc = np.ones((300, 300, 3), dtype=np.uint8) * 240
    valid_doc, reason_d, _ = validate_chest_xray(blank_doc)
    assert not valid_doc
    assert "dynamic range" in reason_d.lower()
