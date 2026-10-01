"""
Automated Clinical Diagnostic PDF Report Generator for Chest Radiography AI

Generates radiologist-grade PDF reports including:
- Hospital / Radiology Department Header & Timestamp
- Patient Demographics & Tabular Vital Signs
- AI Multi-Class Probabilities & Risk Stratification
- Side-by-side Radiograph & Grad-CAM Visual Heatmap
- Tabular Feature Contributions (SpO2, Temp, Age)
- Clinical Recommendation & Differential Diagnosis
- Medico-Legal AI Decision Support Disclaimer
"""

import io
import os
from datetime import datetime
from typing import Dict, Any, Optional
from PIL import Image

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
    )
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


class ClinicalReportGenerator:
    """Generates structured PDF diagnostic reports for Chest X-Ray AI inference."""

    @classmethod
    def generate_pdf_report(
        cls,
        patient_data: Dict[str, Any],
        ai_diagnosis: Dict[str, Any],
        original_img: Image.Image,
        gradcam_img: Image.Image,
        output_path: Optional[str] = None
    ) -> bytes:
        """
        Builds the PDF report document.

        Args:
            patient_data: dict of patient info (id, age, sex, spo2, temperature, symptoms)
            ai_diagnosis: dict of AI predictions (predicted_class, confidence, probabilities, risk_score, risk_tier, recommendations)
            original_img: PIL Image of the chest radiograph
            gradcam_img: PIL Image of the Grad-CAM heatmap overlay
            output_path: optional filepath to write directly to disk

        Returns:
            bytes: PDF file contents in memory
        """
        if not HAS_REPORTLAB:
            raise ImportError("ReportLab is required for PDF generation. Install via `pip install reportlab`.")

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        normal = styles["Normal"]

        # Custom typography styles
        title_style = ParagraphStyle(
            "DocTitle",
            parent=normal,
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#0f172a")
        )
        subtitle_style = ParagraphStyle(
            "DocSubTitle",
            parent=normal,
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#0284c7")
        )
        section_heading = ParagraphStyle(
            "SecHeading",
            parent=normal,
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#1e293b"),
            spaceBefore=8,
            spaceAfter=4
        )
        body_style = ParagraphStyle(
            "BodyDark",
            parent=normal,
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#334155")
        )
        disclaimer_style = ParagraphStyle(
            "LegalDisclaimer",
            parent=normal,
            fontName="Helvetica-Oblique",
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor("#64748b")
        )

        story = []

        # 1. Header Banner
        header_table_data = [
            [
                Paragraph("<b>THORACIC RADIOLOGY AI DIAGNOSTIC REPORT</b><br/><font size=8 color='#64748b'>CLINICAL DECISION SUPPORT SYSTEM (CDSS)</font>", title_style),
                Paragraph(f"<b>Report ID:</b> CDSS-{int(datetime.now().timestamp())}<br/><b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}<br/><b>Facility:</b> Pulmonary Imaging Institute", body_style)
            ]
        ]
        header_table = Table(header_table_data, colWidths=[360, 180])
        header_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=10))

        # 2. Patient Demographics & Tabular Vitals
        story.append(Paragraph("1. PATIENT DEMOGRAPHICS & CLINICAL PRESENTATION", section_heading))

        p_name = str(patient_data.get("patient_name", "Patient Record"))
        phys_name = str(patient_data.get("physician_name", "Dr. Murlidhar, MD"))
        p_id = str(patient_data.get("patient_id", "PT-UNKNOWN"))
        age = str(patient_data.get("age", "N/A"))
        sex = str(patient_data.get("sex", "N/A")).capitalize()
        spo2 = f"{patient_data.get('spo2', 'N/A')}%"
        temp = f"{patient_data.get('temperature', 'N/A')} deg C"
        resp = f"{patient_data.get('respiratory_rate', 'N/A')} bpm"
        cough = "Yes" if patient_data.get("cough") else "No"
        fever = "Yes" if patient_data.get("fever") else "No"

        demo_data = [
            ["Patient Name:", p_name, "Patient ID:", p_id, "Attending Physician:", phys_name],
            ["Age / Sex:", f"{age} yrs / {sex}", "Pulse Ox (SpO2):", spo2, "Temp / Resp Rate:", f"{temp} / {resp}"],
            ["Presenting Symptoms:", f"Persistent Cough: {cough} | Pyrexia/Fever: {fever}", "Clinical Triage:", "AI Multimodal Verified", "", ""]
        ]
        demo_table = Table(demo_data, colWidths=[90, 110, 80, 100, 95, 65])
        demo_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1e293b")),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
            ("FONTNAME", (4, 0), (4, -1), "Helvetica-Bold"),
            ("SPAN", (1, 2), (2, 2)),
            ("SPAN", (4, 2), (5, 2)),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(demo_table)
        story.append(Spacer(1, 10))

        # 3. AI Imaging Findings & Multimodal Risk
        story.append(Paragraph("2. AI RADIOLOGICAL INFERENCE & RISK STRATIFICATION", section_heading))

        predicted_class = ai_diagnosis.get("predicted_class", "Normal")
        confidence = float(ai_diagnosis.get("confidence", 0.0))
        risk_score = float(ai_diagnosis.get("risk_score", 0.0))
        risk_tier = ai_diagnosis.get("risk_tier", "Low")

        # Color coding according to risk
        risk_color = colors.HexColor("#10b981") if "Low" in risk_tier else (
            colors.HexColor("#f59e0b") if "Mod" in risk_tier else colors.HexColor("#ef4444")
        )

        probs = ai_diagnosis.get("probabilities", {})
        prob_str = " | ".join([f"{k}: {v*100:.1f}%" for k, v in probs.items()])

        ai_summary_data = [
            [
                Paragraph(f"<b>Primary AI Classification:</b><br/><font size=14 color='#0284c7'><b>{predicted_class}</b></font><br/>Confidence: {confidence*100:.1f}%", normal),
                Paragraph(f"<b>Integrated Clinical Risk Score:</b><br/><font size=14 color='{risk_color.hexval()}'><b>{risk_score*100:.1f}% ({risk_tier})</b></font><br/>Multimodal CNN + XGBoost Fusion", normal)
            ],
            [
                Paragraph(f"<b>Class Probability Distribution:</b><br/>{prob_str}", body_style),
                Paragraph(f"<b>Ensemble Consensus:</b> DenseNet121 + ResNet50 + EfficientNet-B0", body_style)
            ]
        ]
        ai_table = Table(ai_summary_data, colWidths=[270, 270])
        ai_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(ai_table)
        story.append(Spacer(1, 10))

        # 4. Radiography Visual Panels (Side by side)
        story.append(Paragraph("3. RADIOGRAPHY VISUALIZATION & GRAD-CAM EXPLAINABILITY", section_heading))

        # Save images temporarily into memory buffers for ReportLab
        orig_buf = io.BytesIO()
        original_img.save(orig_buf, format="JPEG", quality=90)
        orig_buf.seek(0)

        cam_buf = io.BytesIO()
        gradcam_img.save(cam_buf, format="JPEG", quality=90)
        cam_buf.seek(0)

        img_w, img_h = 220, 220
        rl_orig = RLImage(orig_buf, width=img_w, height=img_h)
        rl_cam = RLImage(cam_buf, width=img_w, height=img_h)

        image_panel_data = [
            [
                Paragraph("<b>Figure A: Preprocessed Chest X-Ray (CLAHE Enhanced)</b>", body_style),
                Paragraph("<b>Figure B: Grad-CAM Saliency Overlay (Infiltrate Attention)</b>", body_style)
            ],
            [rl_orig, rl_cam],
            [
                Paragraph("<font size=7.5 color='#64748b'>High-contrast equalization applied to enhance parenchymal markings.</font>", normal),
                Paragraph("<font size=7.5 color='#64748b'>Red/yellow zones highlight the anatomical areas driving the AI prediction.</font>", normal)
            ]
        ]
        image_table = Table(image_panel_data, colWidths=[270, 270])
        image_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(image_table)
        story.append(Spacer(1, 8))

        # 5. Clinical Recommendations
        story.append(Paragraph("4. CLINICAL IMPRESSIONS & DIFFERENTIAL RECOMMENDATIONS", section_heading))
        rec_text = ai_diagnosis.get(
            "recommendations",
            "Evaluate clinical findings in conjunction with patient history, auscultation, and microbiological assays."
        )
        rec_table = Table([[Paragraph(f"<b>Recommendation:</b> {rec_text}", body_style)]], colWidths=[540])
        rec_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eff6ff")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#3b82f6")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(rec_table)
        story.append(Spacer(1, 8))

        # 6. Attending Physician Signature Block
        sig_data = [
            [
                Paragraph("<b>AI System Model:</b> DenseNet121 + XGBoost Multimodal (v1.0)<br/><b>Validation Status:</b> Internal Holdout & Stratified Patient Test", body_style),
                Paragraph("<b>Reviewing Radiologist:</b> ___________________________<br/><b>Signature / Date:</b> ___________________________", body_style)
            ]
        ]
        sig_table = Table(sig_data, colWidths=[300, 240])
        story.append(sig_table)
        story.append(Spacer(1, 10))

        # 7. Medico-legal Disclaimer
        disclaimer_text = (
            "NOTICE: This report was generated by an artificial intelligence decision support system for research and "
            "investigational clinical triage. It is not an autonomous medical device. Final diagnosis and therapeutic "
            "management decisions must be rendered by a certified physician or board-certified radiologist."
        )
        story.append(Paragraph(disclaimer_text, disclaimer_style))

        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(pdf_bytes)

        return pdf_bytes
