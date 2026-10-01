"""
Comprehensive Clinical Evaluation & Calibration Metrics for Chest Radiography

Metrics:
- Sensitivity (Recall) - Critical: False negatives must be minimized
- Specificity - Clears healthy patients without causing clinical alarm
- Precision & F1-score (Macro and Weighted)
- Multi-class ROC-AUC (OvR) and PR-AUC
- Confusion Matrix
- Expected Calibration Error (ECE) & Reliability Assessment
- Clinical Decision Threshold Optimization (Prioritizing Sensitivity)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve


def compute_expected_calibration_error(
    y_true_one_hot: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Computes Expected Calibration Error (ECE).
    Measures difference between predicted confidence and empirical accuracy.
    ECE = sum_m ( |B_m| / N ) * | acc(B_m) - conf(B_m) |
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    true_labels = np.argmax(y_true_one_hot, axis=1)
    accuracies = (predictions == true_labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(confidences)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            avg_conf_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * np.abs(acc_in_bin - avg_conf_in_bin)

    return float(ece)


def calculate_clinical_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    threshold: float = 0.5
) -> Dict[str, Union[float, Dict, np.ndarray]]:
    """
    Computes diagnostic metrics for clinical decision support:
    - Sensitivity (Recall) per class & overall
    - Specificity per class & overall
    - Precision, F1-Score, Accuracy
    - ROC-AUC (OvR) and Average Precision (PR-AUC)
    - Expected Calibration Error (ECE)
    - Confusion Matrix
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    num_classes = y_prob.shape[1]

    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    y_pred = np.argmax(y_prob, axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Per-class sensitivity and specificity
    per_class_metrics = {}
    sensitivities = []
    specificities = []

    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fn + fp)

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0

        sensitivities.append(sens)
        specificities.append(spec)

        per_class_metrics[name] = {
            "sensitivity_recall": float(sens),
            "specificity": float(spec),
            "precision": float(prec),
            "f1_score": float(f1),
            "support": int(tp + fn)
        }

    # One-hot encoding for multi-class ROC-AUC & Calibration
    y_true_one_hot = np.zeros_like(y_prob)
    for idx, val in enumerate(y_true):
        y_true_one_hot[idx, val] = 1.0

    try:
        macro_roc_auc = float(roc_auc_score(y_true_one_hot, y_prob, multi_class="ovr", average="macro"))
        weighted_roc_auc = float(roc_auc_score(y_true_one_hot, y_prob, multi_class="ovr", average="weighted"))
    except Exception:
        macro_roc_auc = 0.0
        weighted_roc_auc = 0.0

    try:
        pr_auc = float(average_precision_score(y_true_one_hot, y_prob, average="macro"))
    except Exception:
        pr_auc = 0.0

    ece = compute_expected_calibration_error(y_true_one_hot, y_prob)

    results = {
        "macro_sensitivity": float(np.mean(sensitivities)),
        "macro_specificity": float(np.mean(specificities)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_roc_auc": macro_roc_auc,
        "weighted_roc_auc": weighted_roc_auc,
        "pr_auc": pr_auc,
        "expected_calibration_error_ece": ece,
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist()
    }

    return results


def tune_decision_threshold(
    y_true_binary: np.ndarray,
    y_prob_positive: np.ndarray,
    target_sensitivity: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calibrates operational decision threshold to guarantee clinical screening safety:
    Finds the highest threshold that maintains at least `target_sensitivity` (e.g. 95%),
    maximizing specificity without dropping pneumonia cases.

    Returns:
        optimal_threshold, achieved_sensitivity, achieved_specificity
    """
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_prob_positive)
    # tpr is sensitivity; find index where tpr >= target_sensitivity
    eligible_indices = np.where(tpr >= target_sensitivity)[0]

    if len(eligible_indices) == 0:
        idx = np.argmax(tpr)
    else:
        # Highest threshold among eligible (least false alarms while retaining recall)
        idx = eligible_indices[0]

    opt_threshold = float(thresholds[idx])
    # Clamp threshold between 0.05 and 0.95
    opt_threshold = float(np.clip(opt_threshold, 0.05, 0.95))

    preds = (y_prob_positive >= opt_threshold).astype(int)
    achieved_sens = float(recall_score(y_true_binary, preds, zero_division=0))

    # Specificity
    tn = np.sum((y_true_binary == 0) & (preds == 0))
    fp = np.sum((y_true_binary == 0) & (preds == 1))
    achieved_spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return opt_threshold, achieved_sens, achieved_spec


def plot_confusion_matrix_figure(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix - Chest Radiography",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Renders a sleek medical confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=ax,
        annot_kws={"size": 12, "weight": "bold"}
    )
    ax.set_title(title, fontsize=13, weight="bold", pad=12)
    ax.set_ylabel("True Pathological Diagnosis", fontsize=11)
    ax.set_xlabel("AI Predicted Class", fontsize=11)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig
