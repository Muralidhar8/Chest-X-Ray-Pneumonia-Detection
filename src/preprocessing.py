"""
Preprocessing Pipeline for Chest X-Ray Images
Includes:
- Grayscale to 3-channel conversion
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Medically realistic augmentations (rotation +/-10 deg, jitter, no vertical flips)
- ImageNet normalization & tensor conversions
"""

import cv2
import numpy as np
import torch
from typing import Tuple, Union, Optional, Dict, Any
from PIL import Image

# ImageNet statistics
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Enhances subtle lung infiltrates and consolidations without blowing out noise.

    Args:
        image: Single-channel (grayscale) or 3-channel (RGB/BGR) uint8 array.
        clip_limit: Threshold for contrast limiting. Default 2.0.
        tile_grid_size: Size of grid for histogram equalization (8x8).

    Returns:
        Enhanced image with same number of channels as input.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(image.shape) == 2:
        return clahe.apply(image)
    elif len(image.shape) == 3:
        if image.shape[2] == 1:
            enhanced = clahe.apply(image[:, :, 0])
            return enhanced[:, :, np.newaxis]
        elif image.shape[2] == 3:
            # Convert to LAB color space, apply CLAHE to Luminance channel only
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return image


def ensure_3_channels(image: np.ndarray) -> np.ndarray:
    """
    Replicates single-channel grayscale chest X-rays to 3 channels (RGB)
    for compatibility with ImageNet-pretrained CNN backbones.
    """
    if len(image.shape) == 2:
        return np.stack([image, image, image], axis=-1)
    elif len(image.shape) == 3:
        if image.shape[2] == 1:
            return np.repeat(image, 3, axis=2)
        elif image.shape[2] == 4:
            # Drop alpha channel if present
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3:
            return image
    raise ValueError(f"Unsupported image shape for conversion: {image.shape}")


def validate_chest_xray(
    image: Union[np.ndarray, Image.Image, str, bytes]
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Input Validation Gatekeeper:
    Restricts model inference exclusively to authentic Thoracic Radiographs (Chest X-Rays).
    Strictly rejects everyday personal photos, selfies, portraits, landscapes,
    and non-medical images.

    Validation Criteria:
    1. Dimension & aspect ratio verification (0.55 - 1.85)
    2. Strict monochrome transmission check (rejects color photos, skin tones, selfies)
    3. Facial portrait / selfie detection (rejects personal photos)
    4. Continuous radiographic attenuation entropy & dynamic range (std >= 22.0)
    5. Bilateral thoracic pulmonary air pocket, skeletal rib architecture & mediastinum attenuation profile

    Returns:
        is_valid: bool (True if valid Chest X-ray, False otherwise)
        reason: str (Diagnostic reason if rejected)
        metrics: Dict (Quantitative measurements)
    """
    # 1. Decode to uint8 RGB
    if isinstance(image, str):
        raw = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        if raw is None:
            return False, "Unable to read image file from disk.", {}
        img_rgb = ensure_3_channels(raw) if len(raw.shape) == 2 else (
            cv2.cvtColor(raw, cv2.COLOR_BGR2RGB) if raw.shape[2] == 3 else cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)
        )
    elif isinstance(image, Image.Image):
        img_rgb = np.array(image.convert("RGB"))
    elif isinstance(image, bytes):
        nparr = np.frombuffer(image, np.uint8)
        decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if decoded is None:
            return False, "Failed to decode image buffer.", {}
        img_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    elif isinstance(image, np.ndarray):
        img_rgb = ensure_3_channels(image)
    else:
        return False, f"Unsupported image type: {type(image)}", {}

    h, w, c = img_rgb.shape
    if h < 128 or w < 128:
        return False, "Image resolution too low for diagnostic radiograph analysis (minimum 128x128 required).", {}

    # 2. Aspect Ratio Check
    aspect = h / float(max(1, w))
    if aspect < 0.55 or aspect > 1.85:
        return False, f"Invalid thoracic aspect ratio ({aspect:.2f}). Chest radiographs require standard thoracic field proportions (0.55 - 1.85).", {"aspect_ratio": round(aspect, 2)}

    # 3. Strict Monochrome Transmission Check (Rejects everyday color photos, selfies, portraits)
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    diff_rg = float(np.mean(np.abs(r.astype(float) - g.astype(float))))
    diff_gb = float(np.mean(np.abs(g.astype(float) - b.astype(float))))
    diff_rb = float(np.mean(np.abs(r.astype(float) - b.astype(float))))

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mean_sat = float(np.mean(hsv[:, :, 1]))
    p90_sat = float(np.percentile(hsv[:, :, 1], 90))

    if diff_rg > 10.0 or diff_gb > 10.0 or diff_rb > 12.0 or mean_sat > 16.0 or p90_sat > 32.0:
        return False, (
            f"Non-radiographic color image detected (Color Saturation: {mean_sat:.1f}%, Chrominance: {diff_rg:.1f}). "
            "Chest radiographs are strictly monochrome transmission studies. Everyday photos, selfies, portraits, "
            "and colored images are strictly prohibited."
        ), {
            "mean_saturation": round(mean_sat, 1),
            "chrominance_diff_rg": round(diff_rg, 1),
            "is_color_photo": True
        }

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # 4. Radiographic Dynamic Range & Continuous Gradient Check
    std_dev = float(np.std(gray))
    if std_dev < 22.0:
        return False, (
            f"Insufficient radiographic dynamic range (std={std_dev:.1f}). "
            "Uniform blank images, solid colors, and low-contrast flat documents are rejected."
        ), {"dynamic_range_std": round(std_dev, 1)}

    # 5. Human Portrait / Face Selfie Detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    min_face_dim = int(min(w, h) * 0.22)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(min_face_dim, min_face_dim))

    for (fx, fy, fw, fh) in faces:
        center_x = fx + fw / 2.0
        if 0.20 * w <= center_x <= 0.80 * w:
            face_roi = gray[fy:fy+fh, fx:fx+fw]
            eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=2)
            if len(eyes) >= 1 or (fw * fh) > (0.15 * w * h):
                return False, (
                    "Human facial portrait / selfie detected. This clinical AI system accepts "
                    "exclusively Thoracic Chest Radiographs. Uploading personal photos or selfies of individuals is strictly disallowed."
                ), {"faces_detected": len(faces), "is_selfie": True}

    # 6. Bilateral Thoracic Cavity, Skeletal Architecture & Mediastinum Profile
    mid_y_start, mid_y_end = int(h * 0.20), int(h * 0.75)
    w_third = max(1, w // 3)
    left_lung = gray[mid_y_start:mid_y_end, :w_third]
    mediastinum = gray[mid_y_start:mid_y_end, w_third:2 * w_third]
    right_lung = gray[mid_y_start:mid_y_end, 2 * w_third:]

    mean_left = float(np.mean(left_lung))
    mean_center = float(np.mean(mediastinum))
    mean_right = float(np.mean(right_lung))

    contrast_gradient = mean_center - min(mean_left, mean_right)
    air_left = float(np.mean(left_lung < 115))
    air_right = float(np.mean(right_lung < 115))

    # Rib structure: Costal margins across lung fields
    sobel_l = float(np.mean(np.abs(cv2.Sobel(left_lung, cv2.CV_64F, 0, 1))))
    sobel_r = float(np.mean(np.abs(cv2.Sobel(right_lung, cv2.CV_64F, 0, 1))))

    if contrast_gradient < 20.0 or air_left < 0.30 or air_right < 0.30 or sobel_l < 8.0 or sobel_r < 8.0:
        return False, (
            "Anatomical verification failed: Image lacks bilateral thoracic lung air cavities, "
            "skeletal rib architecture, and central mediastinal attenuation characteristic of chest radiographs."
        ), {
            "central_gradient": round(contrast_gradient, 1),
            "left_lung_air_ratio": round(air_left, 2),
            "right_lung_air_ratio": round(air_right, 2),
            "rib_edge_left": round(sobel_l, 1),
            "rib_edge_right": round(sobel_r, 1)
        }

    metrics = {
        "aspect_ratio": round(aspect, 2),
        "mean_saturation": round(mean_sat, 1),
        "dynamic_range_std": round(std_dev, 1),
        "contrast_gradient": round(contrast_gradient, 1),
        "modality": "Thoracic Radiography (AP/PA View Verified)"
    }
    return True, "Valid Thoracic Chest Radiograph", metrics


def preprocess_xray_image(
    image: Union[np.ndarray, Image.Image, str, bytes],
    target_size: Tuple[int, int] = (224, 224),
    use_clahe: bool = True,
    clip_limit: float = 2.0
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Full inference preprocessing pipeline:
    1. Read / convert to uint8 RGB array
    2. Optional CLAHE enhancement
    3. Resize to target dimensions
    4. Normalize with ImageNet mean & std
    5. Convert to PyTorch Tensor [1, 3, H, W]

    Returns:
        tensor: Normalized PyTorch tensor ready for model inference [1, 3, H, W]
        original_rgb: Unmodified input as uint8 RGB (for side-by-side display)
        clahe_rgb: CLAHE-enhanced uint8 RGB image
    """
    # 1. Load image
    if isinstance(image, str):
        raw = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Could not load image from: {image}")
        if len(raw.shape) == 2:
            original_rgb = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
        elif raw.shape[2] == 3:
            original_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        elif raw.shape[2] == 4:
            original_rgb = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)
        else:
            original_rgb = ensure_3_channels(raw)
    elif isinstance(image, Image.Image):
        original_rgb = np.array(image.convert("RGB"))
    elif isinstance(image, bytes):
        nparr = np.frombuffer(image, np.uint8)
        decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Failed to decode image from bytes.")
        original_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255.0).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        original_rgb = ensure_3_channels(image)
    else:
        raise TypeError(f"Unsupported image input type: {type(image)}")

    # 2. Apply CLAHE
    if use_clahe:
        clahe_rgb = apply_clahe(original_rgb, clip_limit=clip_limit)
    else:
        clahe_rgb = original_rgb.copy()

    # 3. Resize
    resized = cv2.resize(clahe_rgb, target_size, interpolation=cv2.INTER_AREA)

    # 4. Normalize (0-1 float then ImageNet mean & std)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD

    # 5. Convert to PyTorch Tensor [1, 3, H, W]
    # Transpose HWC -> CHW
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()

    return tensor, original_rgb, clahe_rgb


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a normalized PyTorch tensor [3, H, W] or [1, 3, H, W]
    back to uint8 RGB numpy array [H, W, 3] for visualization & Grad-CAM overlay.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    arr = tensor.detach().cpu().permute(1, 2, 0).numpy()
    arr = (arr * IMAGENET_STD) + IMAGENET_MEAN
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


class MedicalAugmentationPipeline:
    """
    Albumentations/OpenCV medically-realistic augmentations:
    - Rotation: limited to +/-10 degrees (preserves anatomical orientation)
    - Brightness & Contrast jitter (simulates scanner calibration variance)
    - Slight shift & scale (+/-5% shift, +/-10% scale)
    - NO vertical flips (lungs are anatomically superior/inferior oriented)
    - NO horizontal flips by default (preserves cardiac silhouette dextrocardia vs normal levocardia)
    """
    def __init__(self, target_size: Tuple[int, int] = (224, 224), is_train: bool = True):
        self.target_size = target_size
        self.is_train = is_train

        try:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2

            if is_train:
                self.transform = A.Compose([
                    A.Resize(target_size[0], target_size[1]),
                    A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_REFLECT),
                    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=0.4, border_mode=cv2.BORDER_REFLECT),
                    A.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
                    ToTensorV2(),
                ])
            else:
                self.transform = A.Compose([
                    A.Resize(target_size[0], target_size[1]),
                    A.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
                    ToTensorV2(),
                ])
            self.use_albumentations = True
        except ImportError:
            self.use_albumentations = False

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """
        Takes uint8 RGB image [H, W, 3] and returns normalized PyTorch tensor [3, H, W].
        """
        # Ensure 3 channels
        image = ensure_3_channels(image)

        if self.use_albumentations:
            augmented = self.transform(image=image)
            return augmented["image"]
        else:
            # Fallback pure OpenCV + NumPy
            resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
            norm = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            tensor = torch.from_numpy(norm).permute(2, 0, 1).float()
            return tensor
