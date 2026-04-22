import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
import os

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Cattle Disease Detection",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap" rel="stylesheet">

<style>
/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #0a1a0e !important;
    color: #ddefd0 !important;
}
.stApp { background: #0a1a0e; }
.block-container { padding: 0 2rem 4rem !important; max-width: 960px !important; }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Hero banner ── */
.hero-wrap {
    background: rgba(29,158,117,0.06);
    border: 0.5px solid rgba(93,202,165,0.15);
    border-radius: 24px;
    padding: 3rem 2rem 2.5rem;
    text-align: center;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute; inset: 0;
    background: repeating-linear-gradient(
        45deg, transparent, transparent 24px,
        rgba(93,202,165,0.025) 24px, rgba(93,202,165,0.025) 25px
    );
    pointer-events: none;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(29,158,117,0.18);
    border: 0.5px solid rgba(93,202,165,0.3);
    color: #5DCAA5;
    font-size: 11px; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 5px 16px; border-radius: 100px;
    margin-bottom: 1.4rem;
}
.pulse-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #5DCAA5; display: inline-block;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100%{opacity:1;transform:scale(1)}
    50%{opacity:0.4;transform:scale(1.5)}
}
.hero-title {
    font-family: 'Syne', sans-serif !important;
    font-size: clamp(2rem, 4vw, 3rem);
    font-weight: 800 !important;
    color: #f2fbe8 !important;
    line-height: 1.1;
    letter-spacing: -0.025em;
    margin-bottom: 1rem;
}
.hero-title em { font-style: normal; color: #5DCAA5; }
.hero-sub {
    font-size: 14px; color: #7a9e72;
    max-width: 560px; margin: 0 auto;
    line-height: 1.75;
}

/* ── Section label ── */
.sec-label {
    font-family: 'Syne', sans-serif !important;
    font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.14em;
    color: #5DCAA5; margin-bottom: 1rem;
    display: flex; align-items: center; gap: 10px;
    margin-top: 2rem;
}
.sec-label::after {
    content: ''; flex: 1; height: 0.5px;
    background: rgba(93,202,165,0.2);
}

/* ── KPI cards ── */
.kpi-card {
    background: rgba(255,255,255,0.04);
    border: 0.5px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 1.1rem 1rem;
    text-align: center; transition: border-color 0.2s;
}
.kpi-card:hover { border-color: rgba(93,202,165,0.3); }
.kpi-val {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.8rem; font-weight: 800;
    color: #5DCAA5; line-height: 1;
}
.kpi-lbl {
    font-size: 11px; color: #527a4a;
    text-transform: uppercase; letter-spacing: 0.07em;
    margin-top: 5px;
}

/* ── Disease cards ── */
.disease-card {
    border-radius: 18px; padding: 1.3rem 1.4rem;
    margin-bottom: 14px;
}
.disease-card.fmd { background: rgba(163,45,45,0.12); border: 0.5px solid rgba(240,149,123,0.2); }
.disease-card.lsd { background: rgba(186,117,23,0.12); border: 0.5px solid rgba(250,199,117,0.2); }
.disease-card.hlt { background: rgba(59,109,17,0.15); border: 0.5px solid rgba(151,196,89,0.25); }
.disease-header {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 0.8rem;
}
.d-dot { width: 11px; height: 11px; border-radius: 50%; flex-shrink: 0; }
.fmd .d-dot { background: #F09595; }
.lsd .d-dot { background: #FAC775; }
.hlt .d-dot { background: #97C459; }
.d-name {
    font-family: 'Syne', sans-serif !important;
    font-size: 14px; font-weight: 700;
}
.fmd .d-name { color: #F7C1C1; }
.lsd .d-name { color: #FAC775; }
.hlt .d-name { color: #C0DD97; }
.d-tag {
    font-size: 10px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em;
    padding: 2px 10px; border-radius: 100px;
    margin-left: auto;
}
.fmd .d-tag { background: rgba(163,45,45,0.25); color: #F09595; }
.lsd .d-tag { background: rgba(186,117,23,0.25); color: #FAC775; }
.hlt .d-tag { background: rgba(59,109,17,0.3); color: #97C459; }
.d-desc { font-size: 13px; color: #8aab7f; line-height: 1.65; margin-bottom: 0.9rem; }
.impact-row { display: flex; gap: 6px; flex-wrap: wrap; }
.impact-pill {
    font-size: 11px; padding: 3px 10px;
    border-radius: 100px;
    border: 0.5px solid rgba(255,255,255,0.1);
    color: #7a9070;
}

/* ── Flow steps ── */
.flow-step {
    display: flex; gap: 14px;
    align-items: flex-start;
    margin-bottom: 12px;
}
.step-num {
    width: 36px; height: 36px; border-radius: 50%;
    background: rgba(29,158,117,0.2);
    border: 0.5px solid rgba(93,202,165,0.35);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Syne', sans-serif !important;
    font-size: 13px; font-weight: 800; color: #5DCAA5;
    flex-shrink: 0;
}
.step-body {
    background: rgba(255,255,255,0.03);
    border: 0.5px solid rgba(255,255,255,0.06);
    border-radius: 14px; padding: 0.9rem 1rem; flex: 1;
}
.step-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 13px; font-weight: 700; color: #c8e8b0;
    margin-bottom: 4px;
}
.step-desc { font-size: 12px; color: #7a9070; line-height: 1.6; }

/* ── Model arch ── */
.model-card {
    background: rgba(255,255,255,0.04);
    border: 0.5px solid rgba(255,255,255,0.08);
    border-radius: 20px; padding: 1.5rem;
    margin-bottom: 1rem;
}
.arch-steps { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; margin: 0.8rem 0 1.2rem; }
.arch-node {
    background: rgba(29,158,117,0.15);
    border: 0.5px solid rgba(93,202,165,0.25);
    border-radius: 10px; padding: 6px 12px;
    font-size: 11px; font-weight: 600; color: #9DD97E;
}
.arch-arrow { color: #3b6d30; font-size: 14px; }
.model-spec-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.model-cell {
    background: rgba(0,0,0,0.2); border-radius: 12px;
    padding: 0.8rem 1rem;
}
.mc-label {
    font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.1em;
    color: #527a4a; margin-bottom: 4px;
}
.mc-val {
    font-family: 'Syne', sans-serif !important;
    font-size: 13px; font-weight: 700; color: #c8e8b0;
}

/* ── Real-world cards ── */
.rw-card {
    background: rgba(255,255,255,0.03);
    border-left: 3px solid rgba(93,202,165,0.35);
    border-radius: 0 14px 14px 0;
    padding: 1rem 1.1rem; margin-bottom: 10px;
}
.rw-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 12px; font-weight: 700;
    color: #9DD97E; margin-bottom: 4px;
}
.rw-desc { font-size: 12px; color: #567050; line-height: 1.6; }

/* ── Benefit cards ── */
.benefit-card {
    background: rgba(255,255,255,0.03);
    border: 0.5px solid rgba(255,255,255,0.06);
    border-radius: 16px; padding: 1.1rem;
    margin-bottom: 10px;
}
.b-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 12px; font-weight: 700;
    color: #c8e8b0; margin-bottom: 4px;
}
.b-desc { font-size: 11px; color: #5f7a57; line-height: 1.55; }

/* ── Result cards ── */
.result-healthy {
    background: rgba(59,109,17,0.2);
    border: 1px solid rgba(99,153,34,0.3);
    border-radius: 20px; padding: 1.5rem;
    margin-bottom: 14px;
}
.result-diseased {
    background: rgba(163,45,45,0.15);
    border: 1px solid rgba(226,75,74,0.25);
    border-radius: 20px; padding: 1.5rem;
    margin-bottom: 14px;
}
.result-label {
    font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.1em;
    opacity: 0.6; margin-bottom: 3px;
}
.result-name-healthy {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.6rem; font-weight: 800; color: #97C459;
}
.result-name-diseased {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.6rem; font-weight: 800; color: #F7C1C1;
}
.action-box {
    background: rgba(186,117,23,0.13);
    border: 0.5px solid rgba(250,199,117,0.2);
    border-radius: 12px; padding: 0.8rem 1rem;
    font-size: 12px; color: #FAC775; line-height: 1.6;
    margin-top: 0.8rem;
}
.scores-wrap {
    background: rgba(255,255,255,0.04);
    border: 0.5px solid rgba(255,255,255,0.07);
    border-radius: 18px; padding: 1.3rem;
    margin-bottom: 14px;
}

/* ── Streamlit component overrides ── */
div[data-testid="stFileUploader"] {
    background: rgba(29,158,117,0.04) !important;
    border: 1.5px dashed rgba(93,202,165,0.35) !important;
    border-radius: 20px !important;
    padding: 1rem !important;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(93,202,165,0.65) !important;
    background: rgba(29,158,117,0.09) !important;
}
.stButton > button {
    background: linear-gradient(135deg,#1D9E75,#3B6D11) !important;
    color: #e8fce0 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important; font-weight: 800 !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important;
    border: none !important; border-radius: 14px !important;
    padding: 0.8rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
div[data-testid="stImage"] img { border-radius: 14px; }
.stProgress > div > div > div > div {
    background: linear-gradient(90deg,#1D9E75,#97C459) !important;
}
.stSpinner > div { border-top-color: #5DCAA5 !important; }

/* ── Footer ── */
.footer-txt {
    text-align: center;
    font-size: 11px; color: #2d4a28;
    text-transform: uppercase; letter-spacing: 0.08em;
    padding-top: 2rem;
    border-top: 0.5px solid rgba(255,255,255,0.05);
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-badge"><span class="pulse-dot"></span>&nbsp;PashuSetu</div>
  <div class="hero-title">Cattle Disease<br><em>Detection System</em></div>
  <p class="hero-sub">
    An AI-powered visual diagnostic tool that identifies Foot-and-Mouth Disease and
    Lumpy Skin Disease from cattle images — enabling faster intervention and better herd management.
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  KPI ROW
# ─────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
for col, val, lbl in zip(
    [k1, k2, k3, k4],
    ["3", "224px", "ONNX", "<1s"],
    ["Disease classes", "Input size", "Model format", "Inference time"],
):
    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-val">{val}</div>
      <div class="kpi-lbl">{lbl}</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  DISEASE DEFINITIONS
# ─────────────────────────────────────────────
st.markdown('<div class="sec-label">Disease definitions</div>', unsafe_allow_html=True)

st.markdown("""
<div class="disease-card fmd">
  <div class="disease-header">
    <span class="d-dot"></span>
    <span class="d-name">Foot-and-Mouth Disease (FMD)</span>
    <span class="d-tag">High risk</span>
  </div>
  <div class="d-desc">
    A highly contagious viral disease caused by an Aphthovirus affecting cloven-hoofed animals.
    Spreads rapidly through direct contact, aerosol, contaminated feed and fomites. Clinical signs
    include fever, excessive salivation and painful blisters on the mouth, feet and teats —
    leading to lameness and severe production loss.
  </div>
  <div class="impact-row">
    <span class="impact-pill">Severe lameness</span>
    <span class="impact-pill">Weight loss</span>
    <span class="impact-pill">Milk drop</span>
    <span class="impact-pill">Trade bans</span>
    <span class="impact-pill">High mortality in young stock</span>
  </div>
</div>

<div class="disease-card lsd">
  <div class="disease-header">
    <span class="d-dot"></span>
    <span class="d-name">Lumpy Skin Disease (LSD)</span>
    <span class="d-tag">Moderate–high risk</span>
  </div>
  <div class="d-desc">
    A viral disease caused by LSDV, a member of the Poxviridae family. Transmitted by biting
    insects such as mosquitoes and flies. Characterized by firm nodular lesions (2–5 cm) across
    the skin, fever, swollen lymph nodes and ocular discharge. Significant economic impact
    through hide damage and reduced milk and meat yield.
  </div>
  <div class="impact-row">
    <span class="impact-pill">Skin nodules</span>
    <span class="impact-pill">Reduced fertility</span>
    <span class="impact-pill">Hide damage</span>
    <span class="impact-pill">Abortion risk</span>
    <span class="impact-pill">Secondary infections</span>
  </div>
</div>

<div class="disease-card hlt">
  <div class="disease-header">
    <span class="d-dot"></span>
    <span class="d-name">Healthy cattle</span>
    <span class="d-tag">Clear</span>
  </div>
  <div class="d-desc">
    Healthy cattle show clear, smooth skin with no lesions, blistering, nodules or abnormal
    discharge. Normal posture, appetite and gait are preserved. Regular screening establishes
    a baseline — enabling faster detection of deviations when disease begins to emerge.
  </div>
  <div class="impact-row">
    <span class="impact-pill">Normal gait</span>
    <span class="impact-pill">Clear skin</span>
    <span class="impact-pill">Good appetite</span>
    <span class="impact-pill">Optimal milk yield</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HOW THE MODEL WORKS
# ─────────────────────────────────────────────
st.markdown('<div class="sec-label">How the model works</div>', unsafe_allow_html=True)

steps = [
    ("Image capture",
     "A photograph of the cattle is taken — ideally showing the skin, face or body area with visible symptoms. Any device camera works; resolution of 224×224 px or above is sufficient."),
    ("Pre-processing",
     "The image is resized to 224×224 pixels, normalized to a 0–1 float range and converted to a 4D tensor (batch × height × width × channels) compatible with the ONNX model input layer."),
    ("CNN feature extraction",
     "The convolutional neural network applies learned filters across the image — detecting low-level patterns (edges, textures) in early layers and high-level features (nodule shapes, lesion borders, color irregularities) in deeper layers."),
    ("Classification head",
     "Extracted features are flattened and passed through fully-connected layers. A final softmax layer converts raw logits into probability scores across the three disease classes, summing to 100%."),
    ("Result output",
     "The class with the highest probability is returned as the prediction, alongside a confidence score and actionable recommendation — all in under one second."),
]

for i, (title, desc) in enumerate(steps, 1):
    st.markdown(f"""
    <div class="flow-step">
      <div class="step-num">{i}</div>
      <div class="step-body">
        <div class="step-title">{title}</div>
        <div class="step-desc">{desc}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MODEL ARCHITECTURE
# ─────────────────────────────────────────────
st.markdown('<div class="sec-label">Model architecture</div>', unsafe_allow_html=True)
st.markdown("""
<div class="model-card">
  <div class="arch-steps">
    <div class="arch-node">Input 224×224×3</div>
    <div class="arch-arrow">→</div>
    <div class="arch-node">Conv layers</div>
    <div class="arch-arrow">→</div>
    <div class="arch-node">Pooling</div>
    <div class="arch-arrow">→</div>
    <div class="arch-node">Dense layers</div>
    <div class="arch-arrow">→</div>
    <div class="arch-node">Softmax × 3</div>
  </div>
  <div class="model-spec-grid">
    <div class="model-cell"><div class="mc-label">Format</div><div class="mc-val">ONNX Runtime</div></div>
    <div class="model-cell"><div class="mc-label">Classes</div><div class="mc-val">FMD · LSD · Healthy</div></div>
    <div class="model-cell"><div class="mc-label">Input shape</div><div class="mc-val">1 × 224 × 224 × 3</div></div>
    <div class="model-cell"><div class="mc-label">Output</div><div class="mc-val">Softmax probabilities</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  REAL-WORLD IMPORTANCE
# ─────────────────────────────────────────────
st.markdown('<div class="sec-label">Real-world importance</div>', unsafe_allow_html=True)

rw_left, rw_right = st.columns(2)

rw_items = [
    ("Economic protection",
     "FMD alone causes estimated losses of $6.5B–$21B annually worldwide. Early AI detection dramatically reduces outbreak spread and trade bans."),
    ("Remote farm access",
     "Farmers in rural areas without regular vet access can use this tool on a smartphone to get an immediate preliminary assessment."),
    ("Herd-scale monitoring",
     "Batch processing enables screening of hundreds of animals per day — practically impossible through manual examination alone."),
    ("Food security",
     "Livestock diseases reduce meat and milk production. Preventing outbreaks at farm level directly supports national food supply chains."),
]

for i, (title, desc) in enumerate(rw_items):
    col = rw_left if i % 2 == 0 else rw_right
    col.markdown(f"""
    <div class="rw-card">
      <div class="rw-title">{title}</div>
      <div class="rw-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  BENEFITS
# ─────────────────────────────────────────────
st.markdown('<div class="sec-label">Benefits</div>', unsafe_allow_html=True)

benefits = [
    ("Speed", "Sub-second diagnosis vs hours or days waiting for laboratory results."),
    ("Non-invasive", "Image-only — no blood samples, swabs or physical handling required."),
    ("Cost reduction", "Reduces unnecessary vet callouts and limits costly treatment delays."),
    ("Early detection", "Catches disease at early visual stages before the herd is exposed."),
    ("Any device", "Runs on standard hardware — no GPU or specialist equipment needed."),
    ("Farmer empowerment", "Gives farmers expert-level insight without veterinary training."),
]

bc1, bc2, bc3 = st.columns(3)
for i, (title, desc) in enumerate(benefits):
    col = [bc1, bc2, bc3][i % 3]
    col.markdown(f"""
    <div class="benefit-card">
      <div class="b-title">{title}</div>
      <div class="b-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MODEL LOADER
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "model.onnx")
    if not os.path.exists(path):
        st.error("⚠️ model.onnx not found. Place it in the same directory as app.py")
        st.stop()
    return ort.InferenceSession(path)

session = load_model()
input_name = session.get_inputs()[0].name
CLASSES = ["foot-and-mouth", "healthy", "lumpy"]
CLASS_LABELS = {"foot-and-mouth": "Foot-and-Mouth Disease", "healthy": "Healthy", "lumpy": "Lumpy Skin Disease"}
BAR_COLORS = {"foot-and-mouth": "#E24B4A", "healthy": "#63991A", "lumpy": "#BA7517"}

# ─────────────────────────────────────────────
#  UPLOAD & PREDICT
# ─────────────────────────────────────────────
st.markdown('<div class="sec-label">Upload and analyze</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload a clear photo of the cattle (skin, face or affected area)",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="visible",
)

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption=f"Uploaded: {uploaded.name}", use_column_width=True)

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        run = st.button("Run AI Analysis")

    if run:
        with st.spinner("Extracting visual features…"):
            img_resized = cv2.resize(img_bgr, (224, 224))
            img_input = (img_resized / 255.0).astype(np.float32)
            img_input = np.expand_dims(img_input, axis=0)
            preds = session.run(None, {input_name: img_input})[0][0]

        pred_idx = int(np.argmax(preds))
        pred_class = CLASSES[pred_idx]
        confidence = float(preds[pred_idx])
        is_healthy = pred_class == "healthy"
        label = CLASS_LABELS[pred_class]

        # ── Main result card
        result_class = "result-healthy" if is_healthy else "result-diseased"
        name_class = "result-name-healthy" if is_healthy else "result-name-diseased"
        status_txt = "Status confirmed" if is_healthy else "Disease detected"
        icon = "✅" if is_healthy else "⚠️"

        action_html = ""
        if not is_healthy:
            action_html = """
            <div class="action-box">
              Recommended action: Isolate the animal immediately and contact a licensed
              veterinarian. Do not re-introduce to the herd until cleared.
            </div>
            """

        st.markdown(f"""
        <div class="{result_class}">
          <div class="result-label">{icon} {status_txt}</div>
          <div class="{name_class}">{label}</div>
          <br>
          <div style="display:flex;justify-content:space-between;margin-bottom:6px">
            <span style="font-size:12px;color:rgba(255,255,255,0.45)">Confidence score</span>
            <span style="font-family:'Syne',sans-serif;font-size:13px;font-weight:700;color:#c8e8b0">{confidence:.1%}</span>
          </div>
          {action_html}
        </div>
        """, unsafe_allow_html=True)

        # ── Confidence bar (native Streamlit progress)
        st.progress(confidence)

        # ── All class scores
        st.markdown('<div class="scores-wrap">', unsafe_allow_html=True)
        st.markdown('<div class="sec-label" style="margin-bottom:0.8rem">All class probabilities</div>', unsafe_allow_html=True)

        for i, cls in enumerate(CLASSES):
            score = float(preds[i])
            col_name, col_bar = st.columns([1, 3])
            with col_name:
                st.markdown(
                    f'<span style="font-size:12px;color:#a0bfa0;text-transform:capitalize">{CLASS_LABELS[cls]}</span>',
                    unsafe_allow_html=True,
                )
            with col_bar:
                st.progress(score)
                st.markdown(
                    f'<span style="font-size:11px;color:#6b8a62;font-family:Syne,sans-serif;font-weight:700">{score:.1%}</span>',
                    unsafe_allow_html=True,
                )

        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer-txt">
  AI for smart farming &bull; ONNX model &bull;
  always consult a licensed veterinarian &bull;
  not a substitute for clinical diagnosis
</div>
""", unsafe_allow_html=True)
