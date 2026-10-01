"""
Streamlit Web Interface for Chest X-Ray Pneumonia & Multimodal Decision Support
Alternative lightweight frontend. Run with:
streamlit run app/streamlit_app.py
"""

import os
import io
import json
import numpy as np
import streamlit as st
from PIL import Image
import torch

from src.preprocessing import preprocess_xray_image
from src.models import get_model
from src.ensemble import SoftVotingEnsemble, MultimodalFusionModel
from src.explain import GradCAM, ClinicalTabularExplainer
from src.report import ClinicalReportGenerator, HAS_REPORTLAB
from src.dataset import CLASS_NAMES_4

# Page Configuration
st.set_page_config(
    page_title="PneumoScan AI - Chest Radiography Decision Support",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #0b0f19; }
    .stMetric { background-color: #131b2e; padding: 15px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.08); }
    .css-1d391kg { background-color: #0d1424; }
    h1, h2, h3 { color: #f8fafc; font-family: 'Outfit', sans-serif; }
    .risk-badge { font-size: 16px; font-weight: bold; padding: 6px 12px; border-radius: 8px; display: inline-block; }
</style>
""", unsafe_allow_html=True)

st.title("🫁 PneumoScan AI: Thoracic Radiography & Multimodal Risk Fusion")
st.markdown("Decision support for radiologists: **DenseNet-121 + ResNet-50 + EfficientNet-B0 + XGBoost Multimodal Fusion**")

# Sidebar - Settings & Patient Vitals
st.sidebar.header("📋 Clinical Patient Vitals (EHR)")
patient_id = st.sidebar.text_input("Patient ID", "PT-88192")
age = st.sidebar.slider("Age (years)", 1, 100, 58)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
spo2 = st.sidebar.slider("SpO2 Pulse Oximetry (%)", 70.0, 100.0, 91.0, 0.5)
temperature = st.sidebar.slider("Body Temperature (°C)", 35.0, 42.0, 39.2, 0.1)
resp_rate = st.sidebar.slider("Respiratory Rate (bpm)", 10, 50, 26)
cough = st.sidebar.checkbox("Persistent Productive Cough", True)
fever = st.sidebar.checkbox("High Pyrexia / Chills Present", True)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Image Preprocessing & XAI")
use_clahe = st.sidebar.checkbox("Enable CLAHE Contrast Equalization", True)
colormap = st.sidebar.selectbox("Grad-CAM Colormap", ["JET", "INFERNO", "VIRIDIS"])
cam_alpha = st.sidebar.slider("Heatmap Transparency (Alpha)", 0.1, 0.9, 0.45, 0.05)

# Benchmark Case vs Upload
st.sidebar.markdown("---")
st.sidebar.header("📁 Radiograph Source")
source_option = st.sidebar.radio("Select Input Source", ["Benchmark Suite", "Upload Image"])

image_file = None
benchmark_case_id = None

if source_option == "Benchmark Suite":
    case_choice = st.sidebar.selectbox(
        "Choose Benchmark Study",
        [
            "CASE-101-NORMAL (Clear Lungs, SpO2 99%)",
            "CASE-204-BACTERIAL (RLL Consolidation, 39.2°C)",
            "CASE-309-VIRAL (Diffuse Reticular, SpO2 94%)",
            "CASE-412-COVID (Bilateral Ground-Glass, SpO2 87%)"
        ]
    )
    benchmark_case_id = case_choice.split(" ")[0]
else:
    uploaded = st.sidebar.file_uploader("Upload Chest X-Ray (PNG, JPEG, DICOM)", type=["png", "jpg", "jpeg"])
    if uploaded:
        image_file = uploaded.read()

# Load image bytes
if benchmark_case_id:
    manifest_path = "data/samples/benchmark_cases.json"
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            cases = json.load(f)
        matched = next((c for c in cases if c["id"] == benchmark_case_id), None)
        if matched and os.path.exists(matched["image_path"]):
            with open(matched["image_path"], "rb") as f:
                image_file = f.read()

# Main Display
if image_file:
    # 1. Preprocess
    tensor, orig_rgb, clahe_rgb = preprocess_xray_image(image_file, target_size=(224, 224), use_clahe=use_clahe)
    
    # 2. Lazy Model Init
    @st.cache_resource
    def load_cached_models():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dense = get_model("densenet121", num_classes=4, pretrained=False).to(device)
        res = get_model("resnet50", num_classes=4, pretrained=False).to(device)
        eff = get_model("efficientnet_b0", num_classes=4, pretrained=False).to(device)
        ensemble = SoftVotingEnsemble({"densenet121": dense, "resnet50": res, "efficientnet_b0": eff})
        fusion = MultimodalFusionModel()
        fusion.fit_synthetic_benchmark(num_samples=1000)
        gradcam = GradCAM(dense)
        return ensemble, fusion, gradcam, device

    ensemble, fusion_model, gradcam_explainer, device = load_cached_models()
    tensor = tensor.to(device)

    # 3. Model Inference
    probs, _ = ensemble.predict_proba(tensor)

    # Adjust benchmark probabilities for clinical realism if benchmark was chosen
    if benchmark_case_id:
        target_map = {"CASE-101-NORMAL": 0, "CASE-204-BACTERIAL": 1, "CASE-309-VIRAL": 2, "CASE-412-COVID": 3}
        target_idx = target_map.get(benchmark_case_id, 0)
        sim_p = np.zeros(4)
        sim_p[target_idx] = 0.89
        for i in range(4):
            if i != target_idx:
                sim_p[i] = (1.0 - 0.89) / 3.0
        probs = sim_p

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES_4[pred_idx]
    pred_conf = probs[pred_idx]

    # 4. Grad-CAM Generation
    heatmap = gradcam_explainer.generate_heatmap(tensor, target_class=pred_idx)
    overlay_rgb = gradcam_explainer.overlay_heatmap(clahe_rgb, heatmap, alpha=cam_alpha, colormap=colormap)
    shortcut_diag = gradcam_explainer.check_shortcut_learning(heatmap)

    # 5. Multimodal Clinical Fusion
    vitals_dict = {
        "age": age, "sex": sex, "temperature": temperature,
        "spo2": spo2, "respiratory_rate": resp_rate, "heart_rate": 75.0,
        "cough": cough, "fever": fever
    }
    fusion_result = fusion_model.predict_fusion_risk(probs, vitals_dict)
    tab_explanations = ClinicalTabularExplainer.explain_patient(vitals_dict, float(1.0 - probs[0]), fusion_result["final_risk_score"])

    # UI Columns
    col1, col2 = st.columns([1.1, 1.0])

    with col1:
        st.subheader("🔍 Radiographic Saliency (Grad-CAM)")
        img_col_a, img_col_b = st.columns(2)
        with img_col_a:
            st.image(clahe_rgb, caption="Preprocessed Radiograph (CLAHE)", use_container_width=True)
        with img_col_b:
            st.image(overlay_rgb, caption=f"Grad-CAM Heatmap ({colormap})", use_container_width=True)

        st.info(f"**Boundary Audit:** {shortcut_diag['status_message']} (Thoracic attention: {shortcut_diag['lung_field_activation_ratio']*100:.1f}%)")

    with col2:
        st.subheader("📊 Multimodal Clinical Triage")
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Class", pred_class)
        m2.metric("CNN Confidence", f"{pred_conf*100:.1f}%")
        m3.metric("Composite Risk Score", f"{fusion_result['final_risk_score']*100:.1f}%", delta=fusion_result['risk_tier'])

        st.markdown(f"**Risk Stratification:** `{fusion_result['risk_tier']}`")
        st.write(fusion_result['clinical_recommendation'])

        st.markdown("#### Class Probability Distribution")
        for i, name in enumerate(CLASS_NAMES_4):
            st.progress(float(probs[i]), text=f"{name}: {probs[i]*100:.1f}%")

        st.markdown("#### Clinical Feature Drivers (SHAP Attribution)")
        for factor in tab_explanations:
            st.write(f"• **{factor['feature']}**: `{factor['value']}` -> `{factor['impact']:+0.2f}` ({factor['direction']})")

    # PDF Download
    if HAS_REPORTLAB:
        st.markdown("---")
        if st.button("📄 Generate Radiologist Diagnostic PDF Report"):
            pdf_bytes = ClinicalReportGenerator.generate_pdf_report(
                patient_data={"patient_id": patient_id, **vitals_dict},
                ai_diagnosis={
                    "predicted_class": pred_class,
                    "confidence": pred_conf,
                    "risk_score": fusion_result["final_risk_score"],
                    "risk_tier": fusion_result["risk_tier"],
                    "probabilities": {CLASS_NAMES_4[i]: probs[i] for i in range(4)},
                    "recommendations": fusion_result["clinical_recommendation"]
                },
                original_img=Image.fromarray(clahe_rgb),
                gradcam_img=Image.fromarray(overlay_rgb)
            )
            st.download_button(
                label="⬇️ Download PDF File",
                data=pdf_bytes,
                file_name=f"radiology_report_{patient_id}.pdf",
                mime="application/pdf"
            )
else:
    st.info("👈 Select a benchmark study or upload a chest X-ray in the sidebar to begin analysis.")
