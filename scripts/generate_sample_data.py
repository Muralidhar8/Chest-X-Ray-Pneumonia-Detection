"""
Synthetic Thoracic Radiograph & Clinical Benchmark Data Generator

Generates high-fidelity anatomical chest radiographs representing:
1. Normal (Clear parenchyma, sharp costophrenic angles, normal cardiothoracic ratio)
2. Bacterial Lobar Pneumonia (Dense focal consolidation in right lower lobe)
3. Viral Pneumonia (Diffuse bilateral interstitial peribronchial infiltrates)
4. COVID-19 (Bilateral peripheral ground-glass opacities, subpleural consolidation)

Also creates paired clinical case metadata (Age, Sex, SpO2, Temp, Symptoms, Narrative).
"""

import os
import json
import cv2
import numpy as np
from PIL import Image


def generate_anatomical_chest_xray(case_type: str = "normal", size: int = 512) -> np.ndarray:
    """
    Renders an anatomically grounded thoracic radiograph with rib cage,
    spine, cardiac silhouette, diaphragm, and pathology-specific opacities.
    """
    # Background soft tissue & air density (air is radiolucent = dark)
    canvas = np.zeros((size, size), dtype=np.float32)

    # 1. Mediastinum & Spine column (radiopaque = white/gray)
    spine_x = size // 2
    for y in range(size):
        canvas[y, spine_x - 18:spine_x + 18] += 0.35 + 0.05 * np.sin(y / 15.0)

    # 2. Thoracic cage outline (rib cage)
    y_grid, x_grid = np.ogrid[:size, :size]

    # Bilateral lung fields (Dark radiolucent cavities)
    # Left lung (anatomical right on frontal AP view)
    lung_r_center = (int(size * 0.34), int(size * 0.48))
    lung_l_center = (int(size * 0.66), int(size * 0.48))

    r_mask = ((x_grid - lung_r_center[0]) ** 2 / (size * 0.16) ** 2 +
              (y_grid - lung_r_center[1]) ** 2 / (size * 0.28) ** 2) <= 1.0

    l_mask = ((x_grid - lung_l_center[0]) ** 2 / (size * 0.16) ** 2 +
              (y_grid - lung_l_center[1]) ** 2 / (size * 0.28) ** 2) <= 1.0

    # Fill base lung density
    canvas[r_mask] = 0.18
    canvas[l_mask] = 0.18

    # Ribs (anterior & posterior curves)
    for i in range(7):
        rib_y = int(size * (0.22 + i * 0.075))
        curve = 0.12 * np.sin(np.linspace(0, np.pi, size))
        rib_band = np.abs(y_grid - (rib_y + curve * size * 0.4)) < (3.5 + i * 0.4)
        canvas[rib_band & (r_mask | l_mask)] += 0.22

    # Clavicles (collar bones near apex)
    clav_r = np.abs((y_grid - size * 0.18) + (x_grid - size * 0.30) * 0.18) < 4
    clav_l = np.abs((y_grid - size * 0.18) - (x_grid - size * 0.70) * 0.18) < 4
    canvas[clav_r & (x_grid < spine_x - 20)] += 0.35
    canvas[clav_l & (x_grid > spine_x + 20)] += 0.35

    # 3. Cardiac Silhouette (Heart resides primarily on patient left / viewer right)
    heart_center = (int(size * 0.54), int(size * 0.58))
    heart_mask = ((x_grid - heart_center[0]) ** 2 / (size * 0.14) ** 2 +
                  (y_grid - heart_center[1]) ** 2 / (size * 0.15) ** 2) <= 1.0
    canvas[heart_mask] += 0.45

    # Aortic knob
    aorta_mask = ((x_grid - (spine_x + 20)) ** 2 / 20**2 + (y_grid - int(size * 0.35)) ** 2 / 24**2) <= 1.0
    canvas[aorta_mask] += 0.38

    # 4. Diaphragm Hemidomes (Right dome slightly higher than left dome)
    right_hemidiaphragm = y_grid > (size * 0.72 - 0.08 * size * np.sin((x_grid - size * 0.1) / (size * 0.4) * np.pi))
    left_hemidiaphragm = y_grid > (size * 0.74 - 0.07 * size * np.sin((x_grid - size * 0.5) / (size * 0.4) * np.pi))
    canvas[right_hemidiaphragm & (x_grid < spine_x)] += 0.42
    canvas[left_hemidiaphragm & (x_grid >= spine_x)] += 0.42

    # Normal Bronchovascular markings (delicate branching from hilum)
    hilum_r = (int(size * 0.42), int(size * 0.45))
    hilum_l = (int(size * 0.58), int(size * 0.45))
    for angle in np.linspace(-0.6, 0.6, 8):
        for dist in range(10, int(size * 0.18)):
            xr = int(hilum_r[0] - dist * np.cos(angle))
            yr = int(hilum_r[1] + dist * np.sin(angle))
            xl = int(hilum_l[0] + dist * np.cos(angle))
            yl = int(hilum_l[1] + dist * np.sin(angle))
            if 0 <= yr < size and 0 <= xr < size:
                canvas[yr, xr] += 0.15 * (1.0 - dist / (size * 0.18))
            if 0 <= yl < size and 0 <= xl < size:
                canvas[yl, xl] += 0.15 * (1.0 - dist / (size * 0.18))

    # ==========================================
    # PATHOLOGY-SPECIFIC OPACITIES (Consolidations)
    # ==========================================
    if case_type.lower() == "bacterial":
        # Focal consolidation in right lower lobe (alveolar filling with air bronchograms)
        focal_center = (int(size * 0.32), int(size * 0.60))
        dist_sq = (x_grid - focal_center[0]) ** 2 / (size * 0.11) ** 2 + (y_grid - focal_center[1]) ** 2 / (size * 0.09) ** 2
        focal_opacity = np.exp(-dist_sq * 2.2) * 0.55
        # Add air bronchogram streaks (dark radiolucent tubular branching through dense white consolidation)
        focal_opacity -= 0.12 * np.sin(x_grid / 6.0) * np.sin(y_grid / 6.0)
        canvas += np.clip(focal_opacity, 0.0, 0.55)

    elif case_type.lower() == "viral":
        # Diffuse interstitial reticular markings across both mid and lower lung zones
        noise = np.random.normal(0, 0.12, (size, size))
        interstitial = cv2.GaussianBlur(noise, (9, 9), 2.5) * 1.8
        interstitial[~ (r_mask | l_mask)] = 0.0
        # Peribronchial cuffing
        canvas += np.clip(interstitial, 0.0, 0.35)

    elif case_type.lower() == "covid":
        # Bilateral peripheral ground-glass opacities (GGO) with subpleural and lower lobe predominance
        # Peripheral right subpleural zone
        ggo_r = ((x_grid - size * 0.20) ** 2 / (size * 0.08) ** 2 + (y_grid - size * 0.56) ** 2 / (size * 0.16) ** 2) <= 1.0
        # Peripheral left subpleural zone
        ggo_l = ((x_grid - size * 0.80) ** 2 / (size * 0.08) ** 2 + (y_grid - size * 0.58) ** 2 / (size * 0.16) ** 2) <= 1.0

        canvas[ggo_r] += 0.38 + 0.05 * np.random.normal(0, 0.1, np.sum(ggo_r))
        canvas[ggo_l] += 0.42 + 0.05 * np.random.normal(0, 0.1, np.sum(ggo_l))

    # Add realistic anatomical soft tissue blur & sensor quantum mottle
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0.8)
    mottle = np.random.normal(0, 0.015, (size, size)).astype(np.float32)
    canvas = np.clip(canvas + mottle, 0.0, 1.0)

    # Convert to 8-bit grayscale
    img_uint8 = (canvas * 255.0).astype(np.uint8)
    return img_uint8


def generate_benchmark_suite(output_dir: str = "data/samples"):
    """Creates the full clinical test benchmark with images and paired JSON metadata."""
    os.makedirs(output_dir, exist_ok=True)

    cases = [
        {
            "id": "CASE-101-NORMAL",
            "filename": "patient_0101_normal.png",
            "type": "normal",
            "label": "Normal",
            "label_idx": 0,
            "patient_info": {
                "patient_id": "PT-94021",
                "age": 34,
                "sex": "Female",
                "temperature": 36.8,
                "spo2": 99,
                "respiratory_rate": 15,
                "heart_rate": 72,
                "cough": False,
                "fever": False,
                "clinical_notes": "Routine pre-operative screening for elective cholecystectomy. No dyspnea, no cough. Lungs clear to auscultation bilaterally."
            }
        },
        {
            "id": "CASE-204-BACTERIAL",
            "filename": "patient_0204_bacteria.png",
            "type": "bacterial",
            "label": "Bacterial Pneumonia",
            "label_idx": 1,
            "patient_info": {
                "patient_id": "PT-88192",
                "age": 58,
                "sex": "Male",
                "temperature": 39.2,
                "spo2": 91,
                "respiratory_rate": 26,
                "heart_rate": 104,
                "cough": True,
                "fever": True,
                "clinical_notes": "Acute onset 3-day history of productive cough with purulent rust-colored sputum, right pleuritic chest pain, and rigors. Auscultation reveals bronchial breath sounds and crackles over right lower base."
            }
        },
        {
            "id": "CASE-309-VIRAL",
            "filename": "patient_0309_viral.png",
            "type": "viral",
            "label": "Viral Pneumonia",
            "label_idx": 2,
            "patient_info": {
                "patient_id": "PT-77314",
                "age": 42,
                "sex": "Female",
                "temperature": 38.3,
                "spo2": 94,
                "respiratory_rate": 21,
                "heart_rate": 88,
                "cough": True,
                "fever": True,
                "clinical_notes": "Progressive non-productive dry cough and malaise for 6 days following viral upper respiratory prodrome. Diffuse scattered expiratory wheezes and minimal fine crackles."
            }
        },
        {
            "id": "CASE-412-COVID",
            "filename": "patient_0412_covid.png",
            "type": "covid",
            "label": "COVID-19",
            "label_idx": 3,
            "patient_info": {
                "patient_id": "PT-66520",
                "age": 67,
                "sex": "Male",
                "temperature": 38.8,
                "spo2": 87,
                "respiratory_rate": 28,
                "heart_rate": 112,
                "cough": True,
                "fever": True,
                "clinical_notes": "Day 9 of PCR-confirmed SARS-CoV-2. Acute hypoxemic respiratory failure with dyspnea at rest, tachypnea, and peripheral bilateral desaturation requiring high-flow nasal cannula."
            }
        }
    ]

    manifest = []
    for c in cases:
        img = generate_anatomical_chest_xray(c["type"], size=512)
        filepath = os.path.join(output_dir, c["filename"])
        cv2.imwrite(filepath, img)

        manifest.append({
            "id": c["id"],
            "image_path": filepath,
            "filename": c["filename"],
            "label": c["label"],
            "label_idx": c["label_idx"],
            "patient_info": c["patient_info"]
        })
        print(f"[Synthesized] {c['id']}: {c['filename']} -> {c['label']}")

    manifest_path = os.path.join(output_dir, "benchmark_cases.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Done] Benchmark manifest written to {manifest_path}")
    return manifest


if __name__ == "__main__":
    generate_benchmark_suite()
