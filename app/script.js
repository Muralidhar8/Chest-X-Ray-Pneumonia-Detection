/**
 * PneumoScan AI - Frontend Interactive Controller
 * Handles image rendering, interactive Before/After curtain split slider,
 * Grad-CAM overlays, patient demographics sync, multimodal fusion, and PDF report downloads.
 */

let currentCaseId = "CASE-101-NORMAL";
let currentViewMode = "split";
let cachedGradcamBase64 = null;
let cachedOriginalBase64 = null;
let currentUploadedFile = null;
let lastDiagnosisData = null;
let isDraggingSplit = false;

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
  fetchHealth();
  initSplitSlider();
  loadBenchmarkCase(currentCaseId);
});

// Telemetry check
async function fetchHealth() {
  try {
    const res = await fetch("/api/health");
    if (res.ok) {
      const data = await res.json();
      const devEl = document.getElementById("telemetry-device");
      if (devEl) {
        devEl.textContent = data.cuda_available ? "NVIDIA CUDA (Active)" : "PyTorch 2.8 Engine";
      }
    }
  } catch (err) {
    console.warn("API healthcheck warning:", err);
  }
}

// 1-Click Benchmark Case Loader
async function loadBenchmarkCase(caseId) {
  currentCaseId = caseId;
  currentUploadedFile = null;

  // Update UI card selection
  const cards = document.querySelectorAll(".benchmark-card");
  cards.forEach(card => card.classList.remove("active"));
  
  const targetCard = Array.from(cards).find(c => c.getAttribute("onclick")?.includes(caseId));
  if (targetCard) targetCard.classList.add("active");

  showLoading(true);

  try {
    // 1. Fetch benchmark metadata to populate vitals
    const casesRes = await fetch("/api/benchmark-cases");
    if (casesRes.ok) {
      const { cases } = await casesRes.json();
      const matched = cases.find(c => c.id === caseId);
      if (matched && matched.patient_info) {
        populateVitalsForm(matched.patient_info);
      }
    }

    // 2. Fetch image prediction & Grad-CAM concurrently
    await Promise.all([
      runImageInference(),
      fetchGradCamOverlay()
    ]);

    // 3. Run multimodal fusion with the pre-populated vitals
    await runMultimodalFusion();

  } catch (err) {
    console.error("Error loading benchmark case:", err);
  } finally {
    showLoading(false);
  }
}

function populateVitalsForm(info) {
  const pName = info.patient_name || "Sarah Jenkins";
  const physName = info.physician_name || "Dr. Murlidhar, MD";

  if (document.getElementById("input-patient-name")) {
    document.getElementById("input-patient-name").value = pName;
  }
  if (document.getElementById("input-physician")) {
    document.getElementById("input-physician").value = physName;
  }
  
  document.getElementById("input-patient-id").value = info.patient_id || "PT-UNKNOWN";
  document.getElementById("input-age").value = info.age || 45;
  document.getElementById("input-sex").value = info.sex || "Male";
  document.getElementById("input-spo2").value = info.spo2 || 98;
  document.getElementById("input-temp").value = info.temperature || 37.0;
  document.getElementById("input-resp").value = info.respiratory_rate || 16;
  document.getElementById("input-cough").checked = !!info.cough;
  document.getElementById("input-fever").checked = !!info.fever;

  syncPatientName(pName);

  // Update telemetry heart rate
  const hrEl = document.getElementById("telemetry-hr");
  if (hrEl) hrEl.textContent = info.heart_rate || 72;

  evaluateSpO2Alert(info.spo2);
}

function syncPatientName(val) {
  const name = val || "Patient Record";
  const pId = document.getElementById("input-patient-id")?.value || "PT-UNKNOWN";
  
  const headerName = document.getElementById("header-patient-name");
  if (headerName) headerName.textContent = name;

  const headerId = document.getElementById("header-patient-id");
  if (headerId) headerId.textContent = pId;

  const summaryName = document.getElementById("summary-patient-name");
  if (summaryName) summaryName.textContent = name;
}

function evaluateSpO2Alert(val) {
  const spo2 = parseFloat(val);
  const spo2Input = document.getElementById("input-spo2");
  if (spo2 < 90) {
    spo2Input.style.borderColor = "var(--accent-rose)";
    spo2Input.style.boxShadow = "0 0 10px rgba(244, 63, 94, 0.3)";
  } else if (spo2 < 94) {
    spo2Input.style.borderColor = "var(--accent-amber)";
    spo2Input.style.boxShadow = "0 0 10px rgba(245, 158, 11, 0.3)";
  } else {
    spo2Input.style.borderColor = "";
    spo2Input.style.boxShadow = "";
  }
}

// Image Inference
async function runImageInference() {
  const useClahe = document.getElementById("clahe-toggle").checked;
  const formData = new FormData();

  if (currentUploadedFile) {
    formData.append("file", currentUploadedFile);
  } else if (currentCaseId) {
    formData.append("case_id", currentCaseId);
  }

  formData.append("model_name", "ensemble");
  formData.append("use_clahe", useClahe);
  formData.append("clip_limit", 2.0);

  const res = await fetch("/api/predict/image", { method: "POST", body: formData });
  if (res.ok) {
    const data = await res.json();
    updateClassProbabilities(data.probabilities);
    lastDiagnosisData = data;
  }
}

// Fetch Grad-CAM Overlay
async function fetchGradCamOverlay() {
  const colormap = document.getElementById("gradcam-colormap").value;
  const alpha = parseFloat(document.getElementById("gradcam-alpha").value) / 100.0;

  const formData = new FormData();
  if (currentUploadedFile) {
    formData.append("file", currentUploadedFile);
  } else if (currentCaseId) {
    formData.append("case_id", currentCaseId);
  }
  formData.append("colormap", colormap);
  formData.append("alpha", alpha);

  const res = await fetch("/api/explain/gradcam", { method: "POST", body: formData });
  if (res.ok) {
    const data = await res.json();
    cachedGradcamBase64 = data.gradcam_overlay_base64;
    cachedOriginalBase64 = data.original_clahe_base64;

    renderActiveView();

    // Update shortcut learning audit card
    if (data.shortcut_learning_audit) {
      updateShortcutAudit(data.shortcut_learning_audit);
    }
  }
}

// Multimodal Fusion (EHR vitals + CNN probabilities)
async function runMultimodalFusion() {
  const vitals = {
    patient_name: document.getElementById("input-patient-name")?.value || "Sarah Jenkins",
    patient_id: document.getElementById("input-patient-id").value,
    physician_name: document.getElementById("input-physician")?.value || "Dr. Murlidhar, MD",
    age: parseFloat(document.getElementById("input-age").value),
    sex: document.getElementById("input-sex").value,
    spo2: parseFloat(document.getElementById("input-spo2").value),
    temperature: parseFloat(document.getElementById("input-temp").value),
    respiratory_rate: parseFloat(document.getElementById("input-resp").value),
    heart_rate: 75.0,
    cough: document.getElementById("input-cough").checked,
    fever: document.getElementById("input-fever").checked,
  };

  const formData = new FormData();
  if (currentUploadedFile) {
    formData.append("file", currentUploadedFile);
  } else if (currentCaseId) {
    formData.append("case_id", currentCaseId);
  }
  formData.append("vitals_json", JSON.stringify(vitals));

  try {
    const res = await fetch("/api/predict/multimodal", { method: "POST", body: formData });
    if (res.ok) {
      const data = await res.json();
      const risk = data.multimodal_risk;
      const imgPred = data.image_prediction;

      // Update Gauge & Tiers
      updateRiskMeter(risk.final_risk_score, risk.risk_tier, risk.risk_color);
      document.getElementById("predicted-diagnosis").textContent = imgPred.predicted_class;
      document.getElementById("clinical-recommendation").textContent = risk.clinical_recommendation;

      // Update multi-class probability bars
      updateClassProbabilities(imgPred.probabilities);

      // Update SHAP tabular attributions
      updateShapAttributions(data.clinical_feature_attributions);

      // Cache for PDF export & modal
      lastDiagnosisData = {
        predicted_class: imgPred.predicted_class,
        confidence: imgPred.confidence,
        risk_score: risk.final_risk_score,
        risk_tier: risk.risk_tier,
        probabilities: imgPred.probabilities,
        recommendations: risk.clinical_recommendation
      };
    }
  } catch (err) {
    console.error("Multimodal fusion error:", err);
  }
}

function updateRiskMeter(score, tier, color) {
  const percent = Math.round(score * 1000) / 10;
  document.getElementById("risk-percent").textContent = `${percent}%`;

  const circle = document.getElementById("meter-circle");
  // Circumference: 2 * pi * 42 ~= 263.89
  const totalOffset = 264;
  const strokeOffset = totalOffset - (totalOffset * (percent / 100));
  circle.style.strokeDashoffset = strokeOffset;
  circle.style.stroke = color || "var(--accent-emerald)";

  const tierPill = document.getElementById("risk-tier-pill");
  tierPill.textContent = tier;
  tierPill.className = "risk-tier-badge";
  if (tier.toLowerCase().includes("low") || tier.toLowerCase().includes("normal")) {
    tierPill.classList.add("low-risk");
  } else if (tier.toLowerCase().includes("mod")) {
    tierPill.classList.add("moderate-risk");
  } else if (tier.toLowerCase().includes("high")) {
    tierPill.classList.add("high-risk");
  } else {
    tierPill.classList.add("critical-risk");
  }
}

function updateClassProbabilities(probs) {
  for (const [cls, p] of Object.entries(probs)) {
    const pct = (p * 100).toFixed(1) + "%";
    if (cls.includes("Normal")) {
      document.getElementById("prob-normal").style.width = pct;
      document.getElementById("val-normal").textContent = pct;
    } else if (cls.includes("Bacterial")) {
      document.getElementById("prob-bacterial").style.width = pct;
      document.getElementById("val-bacterial").textContent = pct;
    } else if (cls.includes("Viral")) {
      document.getElementById("prob-viral").style.width = pct;
      document.getElementById("val-viral").textContent = pct;
    } else if (cls.includes("COVID")) {
      document.getElementById("prob-covid").style.width = pct;
      document.getElementById("val-covid").textContent = pct;
    }
  }
}

function updateShapAttributions(factors) {
  const container = document.getElementById("shap-factors-container");
  container.innerHTML = "";

  if (!factors || factors.length === 0) {
    container.innerHTML = '<div class="shap-item">No significant feature deviations detected.</div>';
    return;
  }

  factors.forEach(f => {
    const item = document.createElement("div");
    item.className = "shap-item";

    let impactClass = "neutral";
    if (f.impact < 0) impactClass = "protective";
    else if (f.impact > 0.1) impactClass = "risk-increase";

    const sign = f.impact > 0 ? `+${f.impact}` : `${f.impact}`;
    item.innerHTML = `
      <span class="shap-feature">${f.feature} (${f.value})</span>
      <span class="shap-impact ${impactClass}">${sign} (${f.direction})</span>
    `;
    container.appendChild(item);
  });
}

function updateShortcutAudit(audit) {
  const ratio = Math.round(audit.lung_field_activation_ratio * 100);
  document.getElementById("audit-bar").style.width = `${ratio}%`;

  const badge = document.getElementById("audit-badge");
  const desc = document.getElementById("audit-desc");

  if (audit.shortcut_learning_suspected) {
    badge.className = "audit-badge badge-warn";
    badge.textContent = "Caution: Peripheral Attention";
    desc.innerHTML = `Warning: <strong>${ratio}%</strong> lung activation. Potential text marker or boundary bias detected.`;
  } else {
    badge.className = "audit-badge badge-pass";
    badge.textContent = "Verified Lung Attention";
    desc.innerHTML = `The Grad-CAM gradient attention is concentrated in <strong>${ratio}%</strong> of the thoracic parenchyma, ruling out text marker or border shortcut bias.`;
  }
}

// Interactive Before/After Split Curtain Wipe Slider
function initSplitSlider() {
  const container = document.getElementById("split-container");
  const handle = document.getElementById("split-divider");
  const clip = document.getElementById("split-clip");

  if (!container || !handle || !clip) return;

  function moveSplit(clientX) {
    const rect = container.getBoundingClientRect();
    let x = clientX - rect.left;
    if (x < 15) x = 15;
    if (x > rect.width - 15) x = rect.width - 15;
    const pct = (x / rect.width) * 100;
    handle.style.left = `${pct}%`;
    clip.style.width = `${pct}%`;
  }

  container.addEventListener("mousedown", (e) => {
    isDraggingSplit = true;
    moveSplit(e.clientX);
  });

  window.addEventListener("mousemove", (e) => {
    if (!isDraggingSplit) return;
    moveSplit(e.clientX);
  });

  window.addEventListener("mouseup", () => {
    isDraggingSplit = false;
  });

  // Touch Support
  container.addEventListener("touchstart", (e) => {
    isDraggingSplit = true;
    if (e.touches[0]) moveSplit(e.touches[0].clientX);
  });

  window.addEventListener("touchmove", (e) => {
    if (!isDraggingSplit) return;
    if (e.touches[0]) moveSplit(e.touches[0].clientX);
  });

  window.addEventListener("touchend", () => {
    isDraggingSplit = false;
  });
}

// View Switching
function switchViewMode(mode) {
  currentViewMode = mode;
  const buttons = document.querySelectorAll(".viewport-tabs .tab-btn");
  buttons.forEach(btn => btn.classList.remove("active"));

  const target = document.getElementById(`tab-${mode}`);
  if (target) target.classList.add("active");

  renderActiveView();
}

function renderActiveView() {
  const mainImg = document.getElementById("main-xray-view");
  const splitContainer = document.getElementById("split-container");
  const splitOrig = document.getElementById("split-img-original");
  const splitOverlay = document.getElementById("split-img-overlay");

  if (currentViewMode === "split") {
    if (mainImg) mainImg.classList.add("hidden");
    if (splitContainer) splitContainer.classList.remove("hidden");

    if (splitOrig && cachedOriginalBase64) splitOrig.src = cachedOriginalBase64;
    if (splitOverlay && cachedGradcamBase64) splitOverlay.src = cachedGradcamBase64;
  } else {
    if (splitContainer) splitContainer.classList.add("hidden");
    if (mainImg) mainImg.classList.remove("hidden");

    if (currentViewMode === "overlay" && cachedGradcamBase64) {
      mainImg.src = cachedGradcamBase64;
    } else if (currentViewMode === "original" && cachedOriginalBase64) {
      mainImg.src = cachedOriginalBase64;
    } else if (cachedGradcamBase64) {
      mainImg.src = cachedGradcamBase64;
    }
  }
}

function updateAlpha(val) {
  document.getElementById("alpha-val").textContent = `${val}%`;
  clearTimeout(window.alphaTimeout);
  window.alphaTimeout = setTimeout(() => {
    fetchGradCamOverlay();
  }, 250);
}

function changeColormap(val) {
  fetchGradCamOverlay();
}

function handleImageParamChange() {
  runImageInference();
  fetchGradCamOverlay();
}

function toggleInvert() {
  const isInverted = document.getElementById("invert-toggle").checked;
  const stage = document.getElementById("image-stage");
  if (stage) {
    if (isInverted) stage.classList.add("inverted");
    else stage.classList.remove("inverted");
  }
}

// Rejection Banner Controls
function dismissRejectionBanner() {
  const banner = document.getElementById("rejection-banner");
  if (banner) banner.classList.add("hidden");
}

function showRejectionBanner(message) {
  const banner = document.getElementById("rejection-banner");
  const msgEl = document.getElementById("rejection-msg");
  if (banner && msgEl) {
    msgEl.textContent = message || "This medical AI system accepts ONLY Chest X-Rays. Personal photos, selfies, portraits, and non-radiographic images are strictly prohibited.";
    banner.classList.remove("hidden");
  }
}

// File Upload Handler
function triggerFileInput() {
  document.getElementById("file-input").click();
}

async function handleFileUpload(event) {
  const file = event.target.files ? event.target.files[0] : null;
  if (!file) return;

  dismissRejectionBanner();
  showLoading(true);

  // Send initial prediction request to validate radiograph modality
  const formData = new FormData();
  formData.append("file", file);
  formData.append("model_name", "ensemble");
  formData.append("use_clahe", document.getElementById("clahe-toggle")?.checked ?? true);
  formData.append("clip_limit", 2.0);

  try {
    const res = await fetch("/api/predict/image", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok || data.error === "NON_XRAY_IMAGE_DETECTED") {
      showLoading(false);
      showRejectionBanner(data.message || "Non-radiographic image detected. Please upload an authentic Chest Radiograph.");
      currentUploadedFile = null;
      if (event.target) event.target.value = "";
      return;
    }

    // Valid Chest Radiograph accepted
    currentUploadedFile = file;
    currentCaseId = null;

    document.querySelectorAll(".benchmark-card").forEach(c => c.classList.remove("active"));

    // Patient metadata for uploaded radiograph
    const uploadName = "Anonymous Patient";
    const uploadId = `UPLOAD-${Date.now().toString().slice(-4)}`;
    document.getElementById("input-patient-name").value = uploadName;
    document.getElementById("input-patient-id").value = uploadId;
    syncPatientName(uploadName);

    // Update probabilities from verified prediction
    updateClassProbabilities(data.probabilities);
    lastDiagnosisData = data;

    // Proceed with Grad-CAM & Multimodal Fusion
    await Promise.all([
      fetchGradCamOverlay(),
      runMultimodalFusion()
    ]);
  } catch (err) {
    console.error("Upload processing error:", err);
  } finally {
    showLoading(false);
  }
}

// Drag & drop support
const dropzone = document.getElementById("dropzone");
if (dropzone) {
  ["dragenter", "dragover"].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.style.borderColor = "var(--accent-cyan)";
    });
  });
  ["dragleave", "drop"].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.style.borderColor = "";
    });
  });
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileUpload({ target: { files: e.dataTransfer.files } });
    }
  });
}

function showLoading(show) {
  const loader = document.getElementById("loading-overlay");
  if (loader) {
    if (show) loader.classList.remove("hidden");
    else loader.classList.add("hidden");
  }
}

// Diagnostic Report Preview Modal
function openReportModal() {
  const modal = document.getElementById("report-modal");
  const body = document.getElementById("modal-report-body");
  if (!modal || !body) return;

  const pName = document.getElementById("input-patient-name")?.value || "Sarah Jenkins";
  const pId = document.getElementById("input-patient-id")?.value || "PT-94021";
  const physName = document.getElementById("input-physician")?.value || "Dr. Murlidhar, MD";
  const age = document.getElementById("input-age")?.value || "34";
  const sex = document.getElementById("input-sex")?.value || "Female";
  const spo2 = document.getElementById("input-spo2")?.value || "99";
  const temp = document.getElementById("input-temp")?.value || "36.8";
  const resp = document.getElementById("input-resp")?.value || "15";
  const cough = document.getElementById("input-cough")?.checked ? "Yes" : "No";
  const fever = document.getElementById("input-fever")?.checked ? "Yes" : "No";

  const predClass = lastDiagnosisData?.predicted_class || document.getElementById("predicted-diagnosis")?.textContent || "Normal";
  const riskScore = lastDiagnosisData?.risk_score !== undefined ? `${(lastDiagnosisData.risk_score * 100).toFixed(1)}%` : document.getElementById("risk-percent")?.textContent || "1.8%";
  const riskTier = lastDiagnosisData?.risk_tier || document.getElementById("risk-tier-pill")?.textContent || "Low / Normal";
  const rec = lastDiagnosisData?.recommendations || document.getElementById("clinical-recommendation")?.textContent || "Standard follow-up recommended.";

  body.innerHTML = `
    <div class="modal-report-box">
      <div style="font-size: 11px; text-transform: uppercase; color: var(--accent-cyan); font-weight: 700; margin-bottom: 8px;">
        1. Patient Demographics & Presenting Vitals
      </div>
      <div class="report-field-grid">
        <div class="report-field"><label>Patient Name</label><span>${pName}</span></div>
        <div class="report-field"><label>Patient ID / MRN</label><span>${pId}</span></div>
        <div class="report-field"><label>Attending Physician</label><span>${physName}</span></div>
        <div class="report-field"><label>Age / Sex</label><span>${age} yrs / ${sex}</span></div>
        <div class="report-field"><label>Pulse Ox (SpO2)</label><span>${spo2}%</span></div>
        <div class="report-field"><label>Body Temp / Resp</label><span>${temp}°C / ${resp} bpm</span></div>
        <div class="report-field"><label>Cough / Fever</label><span>${cough} / ${fever}</span></div>
        <div class="report-field"><label>Study Date</label><span>${new Date().toLocaleDateString()}</span></div>
        <div class="report-field"><label>Clinical Facility</label><span>Pulmonary AI Institute</span></div>
      </div>
    </div>

    <div class="modal-report-box">
      <div style="font-size: 11px; text-transform: uppercase; color: var(--accent-cyan); font-weight: 700; margin-bottom: 8px;">
        2. AI Radiological Findings & Multimodal Risk
      </div>
      <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(0,0,0,0.25); padding: 12px; border-radius: 8px; margin-bottom: 10px;">
        <div>
          <div style="font-size: 11px; color: var(--text-muted);">PRIMARY DIAGNOSTIC IMPRESSION</div>
          <div style="font-size: 18px; font-weight: 800; color: var(--accent-blue);">${predClass}</div>
        </div>
        <div style="text-align: right;">
          <div style="font-size: 11px; color: var(--text-muted);">COMPOSITE INFECTION RISK</div>
          <div style="font-size: 18px; font-weight: 800; color: var(--accent-rose);">${riskScore} (${riskTier})</div>
        </div>
      </div>
      <div style="font-size: 12.5px; color: var(--text-secondary); line-height: 1.5;">
        <strong>Clinical Recommendation:</strong> ${rec}
      </div>
    </div>

    <div style="display: flex; gap: 14px; align-items: center; justify-content: center; background: rgba(0,0,0,0.3); padding: 12px; border-radius: 10px;">
      <div style="text-align: center;">
        <img src="${cachedOriginalBase64 || '/static/assets/placeholder_xray.png'}" style="width: 140px; height: 140px; object-fit: contain; border-radius: 8px; border: 1px solid var(--border-glass);" />
        <div style="font-size: 10px; color: var(--text-muted); margin-top: 4px;">Figure A: CLAHE Scan</div>
      </div>
      <div style="text-align: center;">
        <img src="${cachedGradcamBase64 || '/static/assets/placeholder_xray.png'}" style="width: 140px; height: 140px; object-fit: contain; border-radius: 8px; border: 1px solid var(--border-glass);" />
        <div style="font-size: 10px; color: var(--text-muted); margin-top: 4px;">Figure B: Grad-CAM Overlay</div>
      </div>
    </div>
  `;

  modal.classList.remove("hidden");
}

function closeReportModal() {
  const modal = document.getElementById("report-modal");
  if (modal) modal.classList.add("hidden");
}

// Official Diagnostic PDF Report Downloader
async function downloadOfficialReport() {
  const btn = document.getElementById("download-pdf-btn");
  const originalText = btn.innerHTML;
  btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Generating PDF Report...';
  btn.disabled = true;

  try {
    const vitals = {
      patient_name: document.getElementById("input-patient-name")?.value || "Sarah Jenkins",
      patient_id: document.getElementById("input-patient-id").value,
      physician_name: document.getElementById("input-physician")?.value || "Dr. Murlidhar, MD",
      age: parseFloat(document.getElementById("input-age").value),
      sex: document.getElementById("input-sex").value,
      spo2: parseFloat(document.getElementById("input-spo2").value),
      temperature: parseFloat(document.getElementById("input-temp").value),
      respiratory_rate: parseFloat(document.getElementById("input-resp").value),
      cough: document.getElementById("input-cough").checked,
      fever: document.getElementById("input-fever").checked,
    };

    const formData = new FormData();
    if (currentUploadedFile) {
      formData.append("file", currentUploadedFile);
    } else if (currentCaseId) {
      formData.append("case_id", currentCaseId);
    }

    formData.append("vitals_json", JSON.stringify(vitals));
    formData.append("diagnosis_json", JSON.stringify(lastDiagnosisData || {}));

    const res = await fetch("/api/report/pdf", {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      throw new Error(`Report generation failed: ${res.statusText}`);
    }

    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `radiology_report_${vitals.patient_name.replace(/\\s+/g, '_')}_${vitals.patient_id}.pdf`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  } catch (err) {
    alert("Error downloading report: " + err.message);
  } finally {
    btn.innerHTML = originalText;
    btn.disabled = false;
  }
}
