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
    page_title="AI Detection",
    layout="wide"
)

# ── MODERN LIGHT UI (Perplexity Style) ─────────────────────
st.markdown("""
<style>
    body {
        background: linear-gradient(to bottom, #F8FAFC, #EEF2FF);
        color: #111827;
    }

    .center-box {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-top: 80px;
    }

    .title {
        font-size: 42px;
        font-weight: 600;
        margin-bottom: 20px;
        color: #1E293B;
    }

    .input-card {
        background: white;
        padding: 20px;
        border-radius: 16px;
        width: 600px;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.08);
        border: 1px solid #E5E7EB;
    }

    .result-card {
        background: white;
        padding: 20px;
        border-radius: 16px;
        margin-top: 20px;
        width: 600px;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.06);
        border: 1px solid #E5E7EB;
    }

    .ai {
        border-left: 5px solid #EF4444;
    }

    .human {
        border-left: 5px solid #22C55E;
    }

    .score {
        font-size: 36px;
        font-weight: bold;
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

# ── CENTER UI ──────────────────────────────────────────────
st.markdown('<div class="center-box">', unsafe_allow_html=True)

st.markdown('<div class="title">AI Detection</div>', unsafe_allow_html=True)

# MODE SWITCH
mode = st.radio("", ["Text", "Image"], horizontal=True)

# INPUT CARD
st.markdown('<div class="input-card">', unsafe_allow_html=True)

result = None

# ───────── TEXT MODE ─────────
if mode == "Text":
    user_text = st.text_area(
        "",
        height=120,
        placeholder="Ask anything or paste content..."
    )

    if st.button("Analyze", use_container_width=True):
        if user_text.strip():
            with st.spinner("Processing..."):
                tokenizer, text_model = get_text_model()
                result = ensemble_predict(user_text, tokenizer, text_model)
        else:
            st.warning("Enter text first")

# ───────── IMAGE MODE ─────────
if mode == "Image":
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        with st.spinner("Processing image..."):
            image_model = get_image_model()
            result = predict_image(image, image_model)

st.markdown('</div>', unsafe_allow_html=True)

# ───────── RESULT DISPLAY ─────────
if result:
    box_class = "ai" if result["label"] == "AI Generated" else "human"

    st.markdown(f'<div class="result-card {box_class}">', unsafe_allow_html=True)

    st.markdown(f"<h3>{result['label']}</h3>", unsafe_allow_html=True)
    st.markdown(f"<div class='score'>{result['score']}%</div>", unsafe_allow_html=True)
    st.write("AI Likelihood")

    col1, col2 = st.columns(2)

    col1.metric("Confidence", f"{result['confidence']}%")
    
    if "roberta_ai" in result:
        col2.metric("Model Score", f"{result['roberta_ai']*100:.0f}%")
    else:
        col2.metric("Real Probability", f"{round(result['real_probability']*100,2)}%")

    st.progress(result['ai_probability'])

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
