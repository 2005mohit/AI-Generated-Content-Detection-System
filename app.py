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

# ── LIGHT PROFESSIONAL UI ──────────────────────────────────
st.markdown("""
<style>
    body {
        background-color: #F9FAFB;
        color: #111827;
    }

    .header {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .subheader {
        color: #6B7280;
        margin-bottom: 25px;
    }

    .card {
        background: white;
        padding: 25px;
        border-radius: 14px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.06);
        border: 1px solid #E5E7EB;
    }

    .upload-box {
        border: 2px dashed #D1D5DB;
        padding: 40px;
        border-radius: 14px;
        text-align: center;
        transition: 0.3s;
    }

    .upload-box:hover {
        border-color: #2563EB;
        background: #F1F5F9;
    }

    .result-ai {
        background: #FEF2F2;
        border-left: 5px solid #DC2626;
        padding: 20px;
        border-radius: 12px;
    }

    .result-human {
        background: #F0FDF4;
        border-left: 5px solid #16A34A;
        padding: 20px;
        border-radius: 12px;
    }

    .metric {
        background: #F9FAFB;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #E5E7EB;
        text-align: center;
    }

    .big-score {
        font-size: 42px;
        font-weight: 700;
        margin-top: 5px;
    }

    .mode-box {
        background: white;
        padding: 10px;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
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
st.markdown('<div class="header">AI Content Authenticity Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Analyze text and images using trained detection models</div>', unsafe_allow_html=True)

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
                <strong>{result['label']}</strong>
                <div class="big-score">{result['score']}%</div>
                <div>AI Likelihood</div>
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
                <strong>{result['label']}</strong>
                <div class="big-score">{result['score']}%</div>
                <div>AI Likelihood</div>
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
