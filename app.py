import streamlit as st
from PIL import Image
import sys
import os

# ── Path setup ─────────────────────────────────────────────
sys.path.append(os.path.dirname(__file__))
from Pipeline.image_pipeline import load_image_model, predict_image
from Pipeline.text_pipeline import load_text_model, ensemble_predict

# Page Config 
st.set_page_config(
    page_title="AI Content Authenticity Detection System",
    layout="wide"
)

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

    .result-uncertain {
        background: #FFFBEB;
        border-left: 5px solid #D97706;
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

    .img-container {
        width: 100%;
        aspect-ratio: 4 / 3;
        overflow: hidden;
        border-radius: 12px;
        background: #F3F4F6;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .img-container img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        object-position: center;
        display: block;
    }
</style>
""", unsafe_allow_html=True)


# Load Models 
@st.cache_resource
def get_image_model():
    return load_image_model("model/image_model/image_model.pth")

@st.cache_resource
def get_text_model():
    return load_text_model("model/text_model")


# Helper: determine label with uncertain zone 
def get_label_and_style(score: float, confidence: float):
    """
    score      : AI likelihood 0–100
    confidence : model confidence 0–100
    Returns (display_label, box_class)
    """
    if confidence < 60:
        return "⚠️ Uncertain", "result-uncertain"
    elif score > 60:
        return "🤖 AI Generated", "result-ai"
    else:
        return "✅ Human / Real", "result-human"


# HEADER 
st.markdown('<div class="header">AI Content Authenticity Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Analyze text and images using trained detection models</div>', unsafe_allow_html=True)

# MODE SWITCH 
mode = st.radio("", ["Text Analysis", "Image Analysis"], horizontal=True)

st.divider()


# TEXT MODE
if "Text" in mode:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Text Detection")

    user_text = st.text_area("", height=180, placeholder="Paste your content here...")
    analyze   = st.button("Analyze", use_container_width=True)

    if analyze:
        if not user_text.strip():
            st.warning("Please enter text.")
        else:
            with st.spinner("Processing..."):
                tokenizer, text_model = get_text_model()
                result = ensemble_predict(user_text, tokenizer, text_model)

            score      = float(result["score"])
            confidence = float(result["confidence"])
            label, box_class = get_label_and_style(score, confidence)

            st.markdown(f"""
            <div class="{box_class}">
                <strong>{label}</strong>
                <div class="big-score">{score:.2f}%</div>
                <div>AI Likelihood</div>
            </div>
            """, unsafe_allow_html=True)

            if box_class == "result-uncertain":
                st.info("⚠️ Model confidence is below 60% — result may not be reliable. Try providing more text.")

            col1, col2, col3, col4 = st.columns(4)
            col1.markdown(f"<div class='metric'><b>Confidence</b><br>{confidence:.1f}%</div>", unsafe_allow_html=True)
            col2.markdown(f"<div class='metric'><b>RoBERTa</b><br>{result['roberta_ai']*100:.0f}%</div>", unsafe_allow_html=True)
            col3.markdown(f"<div class='metric'><b>Heuristic</b><br>{result['heuristic_ai']:.0f}%</div>", unsafe_allow_html=True)
            col4.markdown(f"<div class='metric'><b>Prediction</b><br>{label}</div>", unsafe_allow_html=True)

            st.progress(result["ai_probability"], text=f"Likelihood: {score:.2f}%")

    st.markdown('</div>', unsafe_allow_html=True)


# IMAGE MODE 
if "Image" in mode:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Image Detection")

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:
            # ── Fixed 4:3 aspect-ratio display ──
            import base64
            from io import BytesIO

            buf = BytesIO()
            image.save(buf, format="JPEG")
            b64 = base64.b64encode(buf.getvalue()).decode()

            st.markdown(f"""
            <div class="img-container">
                <img src="data:image/jpeg;base64,{b64}" alt="Uploaded image">
            </div>
            """, unsafe_allow_html=True)

        with col2:
            with st.spinner("Processing image..."):
                image_model = get_image_model()
                result      = predict_image(image, image_model)

            score      = float(result["score"])
            confidence = float(result["confidence"])
            label, box_class = get_label_and_style(score, confidence)

            st.markdown(f"""
            <div class="{box_class}">
                <strong>{label}</strong>
                <div class="big-score">{score:.2f}%</div>
                <div>AI Likelihood</div>
            </div>
            """, unsafe_allow_html=True)

            if box_class == "result-uncertain":
                st.info("⚠️ Model confidence is below 60% — result may not be reliable. Try a clearer or higher-resolution image.")

            colA, colB, colC = st.columns(3)
            colA.markdown(f"<div class='metric'><b>Confidence</b><br>{confidence:.2f}%</div>", unsafe_allow_html=True)
            colB.markdown(f"<div class='metric'><b>AI Score</b><br>{score:.2f}%</div>", unsafe_allow_html=True)
            colC.markdown(f"<div class='metric'><b>Real Prob</b><br>{round(result['real_probability']*100, 2):.2f}%</div>", unsafe_allow_html=True)

            st.progress(result["ai_probability"], text=f"Likelihood: {score:.2f}%")

    st.markdown('</div>', unsafe_allow_html=True)


# FOOTER 
st.divider()
st.caption("AI Content Authenticity Detection System | Built for analysis and research")
