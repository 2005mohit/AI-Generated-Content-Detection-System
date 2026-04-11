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
    page_title="AI Content Authenticity Detection System",
    layout="wide"
)

# ── PROFESSIONAL MINIMAL UI ────────────────────────────────
st.markdown("""
<style>
    body {
        background-color: #F8FAFC;
        color: #0F172A;
    }

    .header {
        font-size: 30px;
        font-weight: 600;
        margin-bottom: 2px;
    }

    .subheader {
        color: #64748B;
        margin-bottom: 20px;
    }

    .card {
        background: white;
        padding: 22px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
    }

    .result-ai {
        background: #FFF1F2;
        border-left: 4px solid #EF4444;
        padding: 18px;
        border-radius: 10px;
    }

    .result-human {
        background: #F0FDF4;
        border-left: 4px solid #22C55E;
        padding: 18px;
        border-radius: 10px;
    }

    .result-uncertain {
        background: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 18px;
        border-radius: 10px;
    }

    .metric {
        background: #F8FAFC;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        text-align: center;
        font-size: 14px;
    }

    .big-score {
        font-size: 36px;
        font-weight: 600;
        margin-top: 5px;
    }

    .image-box {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 20px;
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
st.markdown('<div class="subheader">Analyze text and images using detection models</div>', unsafe_allow_html=True)

# ── MODE SWITCH ────────────────────────────────────────────
mode = st.radio("", ["Text Analysis", "Image Analysis"], horizontal=True)
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

            label = result["label"]
            if result["confidence"] < 60:
                label = "UNCERTAIN"
                box_class = "result-uncertain"
            else:
                box_class = "result-ai" if label == "AI Generated" else "result-human"

            st.markdown(f"""
            <div class="{box_class}">
                <strong>{label}</strong>
                <div class="big-score">{result['score']}%</div>
                <div>AI Likelihood</div>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)

            col1.markdown(f"<div class='metric'><b>Confidence</b><br>{result['confidence']}%</div>", unsafe_allow_html=True)
            col2.markdown(f"<div class='metric'><b>RoBERTa</b><br>{result['roberta_ai']*100:.0f}%</div>", unsafe_allow_html=True)
            col3.markdown(f"<div class='metric'><b>Heuristic</b><br>{result['heuristic_ai']:.0f}%</div>", unsafe_allow_html=True)
            col4.markdown(f"<div class='metric'><b>Prediction</b><br>{label}</div>", unsafe_allow_html=True)

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

        # ── LEFT: IMAGE ──
        with col1:
            st.markdown('<div class="image-box">', unsafe_allow_html=True)
            st.image(image, width=250)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── RIGHT: RESULT ──
        with col2:
            with st.spinner("Processing image..."):
                image_model = get_image_model()
                result = predict_image(image, image_model)

            label = result["label"]
            if result["confidence"] < 60:
                label = "UNCERTAIN"
                box_class = "result-uncertain"
            else:
                box_class = "result-ai" if label == "AI Generated" else "result-human"

            # ✅ FIXED LOGIC HERE
            if label == "AI Generated":
                score_value = result['score']
                score_label = "AI Likelihood"
            elif label == "UNCERTAIN":
                score_value = result['score']
                score_label = "AI Likelihood"
            else:
                score_value = round(result['real_probability'] * 100, 2)
                score_label = "Real Probability"

            st.markdown(f"""
            <div class="{box_class}">
                <strong>{label}</strong>
                <div class="big-score">{score_value}%</div>
                <div>{score_label}</div>
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
st.caption("AI Content Authenticity Detection System | For Research And Analysis")
