"""
PyTorch Dataset and Leak-Free Data Partitioning for Chest X-Ray Images

Features:
- Patient-level splitting to prevent data leakage (Group-aware Stratification)
- Extraction of patient IDs from Kermany / COVID-19 / NIH filenames
- Stratified 70/15/15 split generation (overriding Kermany's flawed 16-image val set)
- Class weighting and WeightedRandomSampler computation for handling class imbalance
- Multi-class support: [Normal, Bacterial Pneumonia, Viral Pneumonia, COVID-19]
- Binary mapping: [Normal, Pneumonia]
"""

import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from typing import List, Dict, Tuple, Optional, Union
from PIL import Image

from src.preprocessing import apply_clahe, MedicalAugmentationPipeline, ensure_3_channels

CLASS_NAMES_4 = ["Normal", "Bacterial Pneumonia", "Viral Pneumonia", "COVID-19"]
CLASS_NAMES_BINARY = ["Normal", "Pneumonia"]

CLASS_TO_IDX_4 = {name: idx for idx, name in enumerate(CLASS_NAMES_4)}
CLASS_TO_IDX_BINARY = {name: idx for idx, name in enumerate(CLASS_NAMES_BINARY)}


def extract_patient_id(filename: str) -> str:
    """
    Extracts the unique patient identifier from chest X-ray filenames.
    Prevents patient-level data leakage where multiple images of the same
    individual appear in both training and test sets.

    Examples:
    - person123_bacteria_456.jpeg -> person123
    - person45_virus_89.jpeg -> person45
    - NORMAL2-IM-1427-0001.jpeg -> NORMAL2-IM-1427
    - COVID-19 (104).png -> COVID-104
    - patient00023_001.png -> patient00023
    """
    base = os.path.basename(filename)

    # Kermany person format (person123_...)
    kermany_match = re.match(r"(person\d+)", base, re.IGNORECASE)
    if kermany_match:
        return kermany_match.group(1).lower()

    # Normal format (NORMAL2-IM-0315-0001.jpeg)
    normal_match = re.match(r"([A-Z0-9]+-IM-\d+)", base, re.IGNORECASE)
    if normal_match:
        return normal_match.group(1).upper()

    # COVID format COVID-19 (123).png or covid_patient_123.png
    covid_match = re.match(r"(covid[-_]?\d+|covid[-_]?\d+\s*\(\d+\))", base, re.IGNORECASE)
    if covid_match:
        return covid_match.group(1).lower().replace(" ", "")

    # Generic patient format: patient123_01
    generic_match = re.match(r"(patient\d+|subject\d+)", base, re.IGNORECASE)
    if generic_match:
        return generic_match.group(1).lower()

    # Fallback: file prefix before first underscore or hyphen
    prefix = re.split(r"[_\-\s]", os.path.splitext(base)[0])[0]
    return prefix if prefix else os.path.splitext(base)[0]


def patient_stratified_split(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    label_col: str = "label",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partitions the dataset by PATIENT to ensure zero data leakage across splits.
    Stratifies patient groups according to their primary diagnosis.

    Returns:
        train_df, val_df, test_df
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    np.random.seed(random_state)

    # Group dataframe by patient, identify dominant label for each patient
    patient_summary = (
        df.groupby(patient_col)[label_col]
        .agg(lambda x: x.mode().iloc[0] if not x.empty else x.iloc[0])
        .reset_index()
    )

    train_patients = []
    val_patients = []
    test_patients = []

    # Stratified patient sampling per label class
    for label, group in patient_summary.groupby(label_col):
        shuffled = group.sample(frac=1.0, random_state=random_state)
        n = len(shuffled)
        n_train = int(np.round(n * train_ratio))
        n_val = int(np.round(n * val_ratio))

        # Adjust for edge cases with small patient counts
        if n > 2 and (n - (n_train + n_val)) <= 0:
            n_train = max(1, n_train - 1)

        p_train = shuffled.iloc[:n_train][patient_col].tolist()
        p_val = shuffled.iloc[n_train:n_train + n_val][patient_col].tolist()
        p_test = shuffled.iloc[n_train + n_val:][patient_col].tolist()

        train_patients.extend(p_train)
        val_patients.extend(p_val)
        test_patients.extend(p_test)

    train_df = df[df[patient_col].isin(train_patients)].copy().reset_index(drop=True)
    val_df = df[df[patient_col].isin(val_patients)].copy().reset_index(drop=True)
    test_df = df[df[patient_col].isin(test_patients)].copy().reset_index(drop=True)

    return train_df, val_df, test_df


class ChestXRayDataset(Dataset):
    """
    PyTorch Dataset for Chest X-Ray Images.

    Supports:
    - Preprocessing with CLAHE
    - Medically sound data augmentation
    - Patient metadata tracking
    - Linked clinical tabular features (Age, SpO2, Temperature, etc.)
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_col: str = "image_path",
        label_col: str = "label_idx",
        patient_col: str = "patient_id",
        tabular_cols: Optional[List[str]] = None,
        target_size: Tuple[int, int] = (224, 224),
        is_train: bool = True,
        use_clahe: bool = True,
        clip_limit: float = 2.0
    ):
        self.df = dataframe.reset_index(drop=True)
        self.image_col = image_col
        self.label_col = label_col
        self.patient_col = patient_col
        self.tabular_cols = tabular_cols
        self.use_clahe = use_clahe
        self.clip_limit = clip_limit

        self.transform = MedicalAugmentationPipeline(
            target_size=target_size,
            is_train=is_train
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        row = self.df.iloc[idx]
        img_path = row[self.image_col]

        # Load raw image
        raw_img = Image.open(img_path).convert("L")
        np_img = np.array(raw_img)

        # CLAHE local contrast enhancement
        if self.use_clahe:
            np_img = apply_clahe(np_img, clip_limit=self.clip_limit)

        # Replicate grayscale to 3 channels for ImageNet backbones
        rgb_img = ensure_3_channels(np_img)

        # Apply augmentation & normalization
        image_tensor = self.transform(rgb_img)

        label = int(row[self.label_col])
        patient_id = str(row[self.patient_col]) if self.patient_col in row else "UNKNOWN"

        sample = {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "patient_id": patient_id,
            "path": img_path
        }

        # Optional tabular clinical data
        if self.tabular_cols and all(col in row for col in self.tabular_cols):
            tab_values = row[self.tabular_cols].values.astype(np.float32)
            sample["tabular"] = torch.from_numpy(tab_values)

        return sample


def compute_class_weights(labels: Union[List[int], np.ndarray, pd.Series]) -> torch.Tensor:
    """
    Computes inverse class frequencies to weight the loss function,
    penalizing errors on the minority class (e.g. Normal vs 3x Pneumonia).

    Formula: weight[c] = total_samples / (num_classes * count[c])
    """
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    num_classes = len(classes)
    total_samples = len(labels)

    weights = np.zeros(num_classes, dtype=np.float32)
    for c, count in zip(classes, counts):
        weights[c] = total_samples / (num_classes * count)

    return torch.from_numpy(weights).float()


def create_weighted_sampler(labels: Union[List[int], np.ndarray, pd.Series]) -> WeightedRandomSampler:
    """
    Builds a WeightedRandomSampler that samples minority class instances
    with higher probability, balancing mini-batches during training.
    """
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    class_to_weight = {c: 1.0 / count for c, count in zip(classes, counts)}

    sample_weights = np.array([class_to_weight[label] for label in labels], dtype=np.float64)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler
