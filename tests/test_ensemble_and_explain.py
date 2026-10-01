"""Unit tests for Ensemble, Multimodal Fusion, Grad-CAM, and Clinical Report Generator"""
import numpy as np
import torch
from PIL import Image
try:
    import pytest
except ImportError:
    pytest = None

from src.models import CustomBaselineCNN
from src.ensemble import SoftVotingEnsemble, MultimodalFusionModel
from src.explain import GradCAM, ClinicalTabularExplainer
from src.report import ClinicalReportGenerator, HAS_REPORTLAB


def test_soft_voting_ensemble():
    m1 = CustomBaselineCNN(num_classes=4)
    m2 = CustomBaselineCNN(num_classes=4)
    ensemble = SoftVotingEnsemble({"m1": m1, "m2": m2})

    x = torch.randn(1, 3, 224, 224)
    probs, breakdown = ensemble.predict_proba(x)

    assert probs.shape == (4,)
    assert np.isclose(np.sum(probs), 1.0, atol=1e-5)
    assert "m1" in breakdown
    assert "m2" in breakdown


def test_multimodal_fusion_prediction():
    fusion = MultimodalFusionModel()
    fusion.fit_synthetic_benchmark(num_samples=100)

    cnn_probs = np.array([0.05, 0.75, 0.15, 0.05])
    vitals = {
        "age": 62, "sex": "male", "temperature": 39.1,
        "spo2": 89.0, "respiratory_rate": 27, "heart_rate": 105,
        "cough": True, "fever": True
    }

    result = fusion.predict_fusion_risk(cnn_probs, vitals)
    assert "final_risk_score" in result
    assert 0.0 <= result["final_risk_score"] <= 1.0
    assert result["risk_tier"] in ["High Pneumonia Risk", "Critical / Severe Infiltrate", "Moderate / Indeterminate", "Low / Normal"]


def test_gradcam_generation():
    model = CustomBaselineCNN(num_classes=4)
    gradcam = GradCAM(model)

    x = torch.randn(1, 3, 224, 224)
    heatmap = gradcam.generate_heatmap(x, target_class=1)

    assert heatmap.shape == (224, 224)
    assert 0.0 <= np.min(heatmap) <= np.max(heatmap) <= 1.0

    dummy_rgb = np.ones((224, 224, 3), dtype=np.uint8) * 100
    overlay = gradcam.overlay_heatmap(dummy_rgb, heatmap)
    assert overlay.shape == (224, 224, 3)

    audit = gradcam.check_shortcut_learning(heatmap)
    assert "shortcut_learning_suspected" in audit
    gradcam.remove_hooks()


def test_tabular_clinical_explainer():
    vitals = {"spo2": 88.0, "temperature": 39.5, "age": 70, "respiratory_rate": 28}
    explanations = ClinicalTabularExplainer.explain_patient(vitals, 0.8, 0.9)

    assert len(explanations) >= 3
    spo2_item = next(e for e in explanations if "SpO2" in e["feature"])
    assert spo2_item["impact"] > 0
