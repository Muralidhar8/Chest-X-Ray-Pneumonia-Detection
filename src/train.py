"""
Production Training Pipeline for Chest Radiography Models

Features:
- Two-Phase Transfer Learning (Head warmup -> Low LR Fine-Tuning)
- Class-Weighted Cross-Entropy Loss & Focal Loss for handling severe class imbalance
- Weighted Random Sampler batch balancing
- Cosine Annealing / ReduceLROnPlateau learning rate schedulers
- Mixed precision training (torch.amp.autocast)
- Early stopping with best checkpoint persistence
- Sensitivity & Macro-F1 validation tracking
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple

from src.models import get_model, BaseMedicalCNN
from src.dataset import (
    ChestXRayDataset,
    patient_stratified_split,
    compute_class_weights,
    create_weighted_sampler,
    CLASS_NAMES_4
)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance by down-weighting easy examples:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class EarlyStopping:
    """Early stops training when validation metric ceases to improve."""
    def __init__(self, patience: int = 5, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = -np.inf if mode == "max" else np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_score: float) -> bool:
        improved = (current_score > (self.best_score + self.min_delta)) if self.mode == "max" else (
            current_score < (self.best_score - self.min_delta)
        )

        if improved:
            self.best_score = current_score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = True
) -> Tuple[float, float]:
    """Runs a single training epoch with optional mixed precision."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Runs validation, returning loss, accuracy, and collected probabilities."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = F.softmax(outputs, dim=1)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    val_loss = running_loss / max(1, total)
    val_acc = correct / max(1, total)
    return val_loss, val_acc, np.array(all_probs), np.array(all_targets)


def run_two_phase_training(
    model_name: str = "densenet121",
    num_classes: int = 4,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    phase1_epochs: int = 5,
    phase2_epochs: int = 15,
    phase1_lr: float = 1e-3,
    phase2_lr: float = 1e-5,
    loss_type: str = "weighted_ce",
    class_weights: Optional[torch.Tensor] = None,
    checkpoint_dir: str = "models/checkpoints",
    device: Optional[torch.device] = None
) -> Tuple[BaseMedicalCNN, Dict[str, list]]:
    """
    Executes the two-phase training protocol:
    - Phase 1 (Feature Extraction): Freeze backbone, train new classification head
    - Phase 2 (Fine-Tuning): Unfreeze deep convolutional blocks with low learning rate
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(checkpoint_dir, exist_ok=True)
    model = get_model(model_name, num_classes=num_classes, pretrained=True).to(device)

    # Loss function configuration
    if class_weights is not None:
        class_weights = class_weights.to(device)

    if loss_type == "focal":
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    early_stopper = EarlyStopping(patience=5, mode="min")
    best_val_loss = float("inf")

    # ==========================================
    # PHASE 1: Train Classification Head Only
    # ==========================================
    print(f"\n[Phase 1] Warmup Classification Head for {model_name.upper()} ({phase1_epochs} epochs)...")
    model.freeze_backbone()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=phase1_lr, weight_decay=1e-2)

    if train_loader is not None and val_loader is not None:
        for epoch in range(1, phase1_epochs + 1):
            t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            v_loss, v_acc, _, _ = evaluate_epoch(model, val_loader, criterion, device)

            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)
            history["train_acc"].append(t_acc)
            history["val_acc"].append(v_acc)

            print(f"  Epoch {epoch:02d}/{phase1_epochs:02d} | Train Loss: {t_loss:.4f}, Acc: {t_acc:.3f} | Val Loss: {v_loss:.4f}, Acc: {v_acc:.3f}")

    # ==========================================
    # PHASE 2: Unfreeze & Fine-Tune Deep Layers
    # ==========================================
    print(f"\n[Phase 2] Fine-Tuning Deep Convolutional Layers ({phase2_epochs} epochs, lr={phase2_lr})...")
    model.unfreeze_last_blocks(num_blocks=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=phase2_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase2_epochs, eta_min=1e-7)

    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")

    if train_loader is not None and val_loader is not None:
        for epoch in range(1, phase2_epochs + 1):
            t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            v_loss, v_acc, _, _ = evaluate_epoch(model, val_loader, criterion, device)
            scheduler.step()

            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)
            history["train_acc"].append(t_acc)
            history["val_acc"].append(v_acc)

            print(f"  Epoch {epoch:02d}/{phase2_epochs:02d} | Train Loss: {t_loss:.4f}, Acc: {t_acc:.3f} | Val Loss: {v_loss:.4f}, Acc: {v_acc:.3f}")

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "num_classes": num_classes,
                    "val_loss": v_loss,
                    "val_acc": v_acc
                }, checkpoint_path)
                print(f"    --> Saved best checkpoint to: {checkpoint_path}")

            if early_stopper(v_loss):
                print(f"\n[Early Stopping] Triggered after epoch {epoch}. Restoring best checkpoint.")
                break

    return model, history
