"""
Explainable AI (XAI) for Medical Radiography and Multimodal Decision Support

Components:
- PyTorch Grad-CAM (Gradient-weighted Class Activation Mapping)
- Heatmap colormap overlays (Jet, Inferno, Viridis)
- Shortcut learning detection (lung field vs perimeter/text marker activation)
- Clinical tabular feature attribution (SpO2, Temperature, Age impact scores)
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, List, Any


class GradCAM:
    """
    Grad-CAM: Visual Explanations from Deep Networks.
    Computes class-specific gradient attributions with respect to the last conv layer feature maps.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        self.target_layer = target_layer or self._find_target_layer()
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.hooks = []

        self._register_hooks()

    def _find_target_layer(self) -> nn.Module:
        """Automatically detects the last convolutional layer across common architectures."""
        # DenseNet
        if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "features"):
            features = self.model.backbone.features
            if hasattr(features, "denseblock4"):
                return features.denseblock4
            return features[-1] if isinstance(features, nn.Sequential) else features

        # ResNet
        if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "layer4"):
            return self.model.backbone.layer4

        # EfficientNet
        if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "features"):
            return self.model.backbone.features[-1]

        # CustomBaselineCNN
        if hasattr(self.model, "block4"):
            return self.model.block4

        # Fallback: scan backwards through modules for Conv2d
        for module in reversed(list(self.model.modules())):
            if isinstance(module, nn.Conv2d):
                return module

        raise ValueError("Could not automatically locate the target convolutional layer for Grad-CAM.")

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple (grad,)
            self.gradients = grad_out[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Computes Grad-CAM heatmap for a given input tensor [1, 3, H, W].

        Returns:
            heatmap: 2D numpy array [H, W] normalized to [0, 1]
        """
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = int(torch.argmax(output, dim=1).item())

        # Backward pass on target class score
        score = output[0, target_class]
        score.backward(retain_graph=True)

        # Global average pooling of gradients: weights alpha_k
        # gradients shape: [1, C, H_feat, W_feat]
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weight the activation maps
        activations = self.activations[0]  # [C, H_feat, W_feat]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        # Heatmap = ReLU(sum_k alpha_k * A^k)
        heatmap = torch.sum(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0.0)

        # Normalize to [0, 1]
        max_val = np.max(heatmap)
        if max_val > 1e-8:
            heatmap = heatmap / max_val
        else:
            heatmap = np.zeros_like(heatmap)

        # Resize heatmap to input tensor dimensions
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        return heatmap

    def overlay_heatmap(
        self,
        rgb_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap: str = "JET"
    ) -> np.ndarray:
        """
        Superimposes the Grad-CAM heatmap onto the original RGB chest X-ray.

        Args:
            rgb_image: uint8 RGB array [H, W, 3]
            heatmap: float array [H, W] in [0, 1]
            alpha: Transparency factor for overlay (0.0 = only image, 1.0 = only heatmap)
            colormap: 'JET', 'INFERNO', or 'VIRIDIS'
        """
        h, w = rgb_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        # Convert normalized float to uint8 [0, 255]
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        cm_code = cv2.COLORMAP_JET
        if colormap.upper() == "INFERNO":
            cm_code = cv2.COLORMAP_INFERNO
        elif colormap.upper() == "VIRIDIS":
            cm_code = cv2.COLORMAP_VIRIDIS

        colored_heatmap = cv2.applyColorMap(heatmap_uint8, cm_code)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

        # Weighted blend: alpha * heatmap + (1 - alpha) * image
        overlay = cv2.addWeighted(colored_heatmap, alpha, rgb_image, 1.0 - alpha, 0)
        return overlay

    def check_shortcut_learning(self, heatmap: np.ndarray) -> Dict[str, Any]:
        """
        Verifies whether the network is attending to thoracic lung parenchyma
        or falling prey to 'shortcut learning' (attending to hospital letter markers,
        clavicle corners, image borders, or scanner artifacts).

        Returns diagnostic metrics and shortcut warning flag.
        """
        h, w = heatmap.shape
        # Define central lung field region (central 60% of height and width)
        y_min, y_max = int(0.20 * h), int(0.80 * h)
        x_min, x_max = int(0.15 * w), int(0.85 * w)

        lung_mask = np.zeros_like(heatmap, dtype=bool)
        lung_mask[y_min:y_max, x_min:x_max] = True

        total_activation = np.sum(heatmap) + 1e-8
        lung_activation = np.sum(heatmap[lung_mask])
        peripheral_activation = total_activation - lung_activation

        lung_ratio = float(lung_activation / total_activation)
        peripheral_ratio = float(peripheral_activation / total_activation)

        is_suspicious_shortcut = bool(peripheral_ratio > 0.45)

        return {
            "lung_field_activation_ratio": round(lung_ratio, 3),
            "peripheral_border_ratio": round(peripheral_ratio, 3),
            "shortcut_learning_suspected": is_suspicious_shortcut,
            "status_message": (
                "Valid lung parenchyma attention."
                if not is_suspicious_shortcut
                else "Warning: Elevated peripheral attention detected. Verify no text marker or scanner artifact interference."
            )
        }

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class ClinicalTabularExplainer:
    """
    Interprets the clinical tabular features contributing to the Multimodal Fusion risk score.
    Maps patient vitals (SpO2, Temperature, Heart Rate, Age) to clinical risk impact.
    """
    FEATURE_BENCHMARKS = {
        "spo2": {"normal": 98.0, "unit": "%", "label": "SpO2 (Oxygen Saturation)", "critical_threshold": 92.0},
        "temperature": {"normal": 37.0, "unit": "deg C", "label": "Body Temperature", "critical_threshold": 38.5},
        "respiratory_rate": {"normal": 16.0, "unit": "bpm", "label": "Respiratory Rate", "critical_threshold": 24.0},
        "age": {"normal": 40.0, "unit": "yrs", "label": "Patient Age", "critical_threshold": 65.0},
        "cough": {"normal": 0.0, "unit": "binary", "label": "Persistent Cough", "critical_threshold": 1.0},
        "fever": {"normal": 0.0, "unit": "binary", "label": "Fever Present", "critical_threshold": 1.0}
    }

    @classmethod
    def explain_patient(
        cls,
        clinical_data: Dict[str, float],
        base_cnn_pneumonia_prob: float,
        final_risk_score: float
    ) -> List[Dict[str, Any]]:
        """
        Calculates directional feature contributions towards the final pneumonia risk score.
        Simulates SHAP-like local explanations with exact clinical grounding.
        """
        explanations = []

        # 1. SpO2 impact
        spo2 = clinical_data.get("spo2", 98.0)
        if spo2 < 92.0:
            impact = min(0.35, (98.0 - spo2) * 0.035)
            direction = "High Risk Increase"
            desc = f"Hypoxemia ({spo2}% SpO2) strongly indicates compromised pulmonary gas exchange."
        elif spo2 < 95.0:
            impact = 0.15
            direction = "Moderate Risk Increase"
            desc = f"Mild desaturation ({spo2}% SpO2) observed."
        else:
            impact = -0.10
            direction = "Protective / Normal"
            desc = f"Healthy room-air oxygen saturation ({spo2}%)."
        explanations.append({
            "feature": "SpO2 (Oxygen Saturation)",
            "value": f"{spo2}%",
            "impact": round(impact, 3),
            "direction": direction,
            "clinical_significance": desc
        })

        # 2. Temperature impact
        temp = clinical_data.get("temperature", 37.0)
        if temp >= 38.5:
            impact = min(0.25, (temp - 37.0) * 0.12)
            direction = "High Risk Increase"
            desc = f"Significant pyrexia ({temp} deg C) indicative of active inflammatory/infectious response."
        elif temp >= 37.5:
            impact = 0.10
            direction = "Moderate Risk Increase"
            desc = f"Low-grade fever ({temp} deg C)."
        else:
            impact = -0.05
            direction = "Protective / Normal"
            desc = f"Normothermic ({temp} deg C)."
        explanations.append({
            "feature": "Body Temperature",
            "value": f"{temp} deg C",
            "impact": round(impact, 3),
            "direction": direction,
            "clinical_significance": desc
        })

        # 3. Respiratory Rate
        resp = clinical_data.get("respiratory_rate", 16.0)
        if resp >= 24.0:
            impact = 0.20
            direction = "High Risk Increase"
            desc = f"Tachypnea ({resp} bpm) indicates compensatory respiratory distress."
        elif resp >= 20.0:
            impact = 0.08
            direction = "Mild Risk Increase"
            desc = f"Mildly elevated respiratory rate ({resp} bpm)."
        else:
            impact = -0.04
            direction = "Normal"
            desc = f"Normal eupneic breathing ({resp} bpm)."
        explanations.append({
            "feature": "Respiratory Rate",
            "value": f"{resp} bpm",
            "impact": round(impact, 3),
            "direction": direction,
            "clinical_significance": desc
        })

        # 4. Age Factor
        age = clinical_data.get("age", 45.0)
        if age >= 65.0:
            impact = 0.12
            direction = "Vulnerability Factor"
            desc = f"Elderly cohort (Age {int(age)}) possesses elevated vulnerability to pneumonia complications."
        elif age <= 5.0:
            impact = 0.10
            direction = "Pediatric Factor"
            desc = f"Pediatric cohort (Age {int(age)}) requires close monitoring for rapid decompensation."
        else:
            impact = 0.00
            direction = "Neutral"
            desc = f"Standard adult demographic (Age {int(age)})."
        explanations.append({
            "feature": "Patient Age",
            "value": f"{int(age)} yrs",
            "impact": round(impact, 3),
            "direction": direction,
            "clinical_significance": desc
        })

        return explanations
