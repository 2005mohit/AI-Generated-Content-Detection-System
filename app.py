import streamlit as st
from PIL import Image
import sys
import os

# ── Path setup ─────────────────────────────────────────────
sys.path.append(os.path.dirname(__file__))
from Pipeline.image_pipeline import load_image_model, predict_image
from Pipeline.text_pipeline import load_text_model, ensemble_predict

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="AI Detection Dashboard",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

/* Hide Streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stToolbar"] { display: none; }

/* Full page background */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    min-height: 100vh;
}

/* Hero header */
.hero-header {
    text-align: center;
    padding: 48px 20px 32px;
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(168,85,247,0.15));
    border-radius: 24px;
    border: 1px solid rgba(99,102,241,0.3);
    margin-bottom: 32px;
    backdrop-filter: blur(10px);
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
    letter-spacing: -1px;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 16px;
    font-weight: 400;
}

/* Mode toggle styling */
.stRadio > div {
    display: flex;
    justify-content: center;
    gap: 12px;
    background: rgba(255,255,255,0.05);
    padding: 8px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.1);
}

.stRadio label {
    color: #cbd5e1 !important;
    font-weight: 500;
    padding: 8px 24px;
    border-radius: 10px;
    transition: all 0.2s;
}

/* Main card */
.card {
    background: rgba(255,255,255,0.04);
    padding: 32px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(20px);
    box-shadow: 0 25px 50px rgba(0,0,0,0.4);
}

/* Section title */
.section-title {
    font-size: 20px;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Text area */
.stTextArea textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 14px !important;
    color: #e2e8f0 !important;
    font-size: 15px !important;
    padding: 16px !important;
    transition: border 0.2s !important;
}

.stTextArea textarea:focus {
    border-color: rgba(129,140,248,0.6) !important;
    box-shadow: 0 0 0 3px rgba(129,140,248,0.15) !important;
}

/* Analyze button */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    padding: 14px 40px !important;
    border-radius: 14px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all 0.3s !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
    margin-top: 12px !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(99,102,241,0.6) !important;
}

/* Result boxes */
.result-ai {
    background: linear-gradient(135deg, rgba(220,38,38,0.15), rgba(239,68,68,0.08));
    border: 1px solid rgba(220,38,38,0.4);
    border-left: 5px solid #dc2626;
    padding: 28px 32px;
    border-radius: 16px;
    margin: 20px 0;
    backdrop-filter: blur(10px);
}

.result-human {
    background: linear-gradient(135deg, rgba(22,163,74,0.15), rgba(34,197,94,0.08));
    border: 1px solid rgba(22,163,74,0.4);
    border-left: 5px solid #16a34a;
    padding: 28px 32px;
    border-radius: 16px;
    margin: 20px 0;
    backdrop-filter: blur(10px);
}

.result-uncertain {
    background: linear-gradient(135deg, rgba(217,119,6,0.15), rgba(245,158,11,0.08));
    border: 1px solid rgba(217,119,6,0.4);
    border-left: 5px solid #d97706;
    padding: 28px 32px;
    border-radius: 16px;
    margin: 20px 0;
}

.result-label {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #94a3b8;
    margin-bottom: 4px;
}

.result-score {
    font-size: 52px;
    font-weight: 800;
    color: #f1f5f9;
    line-height: 1;
    margin: 4px 0;
}

.result-sublabel {
    font-size: 14px;
    color: #64748b;
    margin-top: 2px;
}

/* Metric cards */
.metric {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    padding: 18px 12px;
    border-radius: 14px;
    text-align: center;
    transition: all 0.2s;
}

.metric:hover {
    background: rgba(255,255,255,0.1);
    border-color: rgba(129,140,248,0.4);
    transform: translateY(-2px);
}

.metric b {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #64748b;
    display: block;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 24px;
    font-weight: 700;
    color: #e2e8f0;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #c084fc) !important;
    border-radius: 10px !important;
    height: 8px !important;
}

.stProgress > div {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    height: 8px !important;
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.08) !important;
    margin: 24px 0 !important;
}

/* Footer */
.footer {
    text-align: center;
    padding: 24px;
    color: #334155;
    font-size: 13px;
    margin-top: 20px;
}

/* Upload zone */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(99,102,241,0.4) !important;
    border-radius: 16px !important;
    background: rgba(99,102,241,0.05) !important;
    padding: 20px !important;
    transition: all 0.2s !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,102,241,0.7) !important;
    background: rgba(99,102,241,0.1) !important;
}

/* Image display */
[data-testid="stImage"] img {
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4) !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #6366f1 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load Models ────────────────────────────────────────────
@st.cache_resource
def get_image_model():
    return load_image_model("model/image_model/image_model.pth")

@st.cache_resource
def get_text_model():
    return load_text_model("model/text_model")

# ── HEADER ─────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🛡️ AI Content Authenticity Detector</div>
    <div class="hero-subtitle">Detect AI-generated text & images using RoBERTa + EfficientNet B0</div>
</div>
""", unsafe_allow_html=True)

# ── MODE SWITCH ────────────────────────────────────────────
mode = st.radio(
    "",
    ["Text Analysis", "Image Analysis"],
    horizontal=True
)

st.divider()

# ───────────────────────── TEXT MODE ───────────────────────
if "Text" in mode:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Text Detection")

    user_text = st.text_area(
        "",
        height=180,
        placeholder="Paste your content here..."
    )

    analyze = st.button("Analyze", use_container_width=True)

    if analyze:
        if not user_text.strip():
            st.warning("Please enter text")
        else:
            with st.spinner("Processing..."):
                tokenizer, text_model = get_text_model()
                result = ensemble_predict(user_text, tokenizer, text_model)

            box_class = "result-ai" if result["label"] == "AI Generated" else "result-human"

            st.markdown(f"""
            <div class="{box_class}">
                <div class="result-label">Detection Result</div>
                <div class="result-score">{result['score']}%</div>
                <div class="result-sublabel">AI Likelihood — <strong>{result['label']}</strong></div>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)

            col1.markdown(f"<div class='metric'><b>Confidence</b><br>{result['confidence']}%</div>", unsafe_allow_html=True)
            col2.markdown(f"<div class='metric'><b>RoBERTa</b><br>{result['roberta_ai']*100:.0f}%</div>", unsafe_allow_html=True)
            col3.markdown(f"<div class='metric'><b>Heuristic</b><br>{result['heuristic_ai']:.0f}%</div>", unsafe_allow_html=True)
            col4.markdown(f"<div class='metric'><b>Prediction</b><br>{result['label']}</div>", unsafe_allow_html=True)

            st.progress(result['ai_probability'], text=f"Likelihood: {result['score']}%")

    st.markdown('</div>', unsafe_allow_html=True)


# ───────────────────────── IMAGE MODE ──────────────────────
if "Image" in mode:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Image Detection")

    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns([1,1])

        with col1:
            st.image(image, use_container_width=True)

        with col2:
            with st.spinner("Processing image..."):
                image_model = get_image_model()
                result = predict_image(image, image_model)

            box_class = "result-ai" if result["label"] == "AI Generated" else "result-human"

            st.markdown(f"""
            <div class="{box_class}">
                <div class="result-label">Detection Result</div>
                <div class="result-score">{result['score']}%</div>
                <div class="result-sublabel">AI Likelihood — <strong>{result['label']}</strong></div>
            </div>
            """, unsafe_allow_html=True)

            colA, colB, colC = st.columns(3)

            colA.markdown(f"<div class='metric'><b>Confidence</b><br>{result['confidence']}%</div>", unsafe_allow_html=True)
            colB.markdown(f"<div class='metric'><b>AI Score</b><br>{result['score']}%</div>", unsafe_allow_html=True)
            colC.markdown(f"<div class='metric'><b>Real Probability</b><br>{round(result['real_probability']*100,2)}%</div>", unsafe_allow_html=True)

            st.progress(result['ai_probability'], text=f"Likelihood: {result['score']}%")

    st.markdown('</div>', unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────
st.divider()
st.caption("AI Detection System | Built for analysis and research")
