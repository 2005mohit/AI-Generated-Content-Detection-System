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

# ── LIGHT UI ───────────────────────────────────────────────
st.markdown("""
<style>
    body {
        background-color: #F9FAFB;
        color: #111827;
    }

    .main-title {
        font-size: 28px;
        font-weight: 600;
        margin-bottom: 20px;
    }

    .input-container {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.05);
    }

    .result-box {
        background: white;
        border-radius: 16px;
        padding: 20px;
        margin-top: 20px;
        border: 1px solid #E5E7EB;
    }

    .ai {
        border-left: 5px solid #DC2626;
    }

    .human {
        border-left: 5px solid #16A34A;
    }

    .score {
        font-size: 32px;
        font-weight: bold;
    }

    .sidebar-title {
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODELS ────────────────────────────────────────────
@st.cache_resource
def get_image_model():
    return load_image_model("model/image_model/image_model.pth")

@st.cache_resource
def get_text_model():
    return load_text_model("model/text_model")

# ── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">History</div>', unsafe_allow_html=True)
    st.write("ai content authenticity system")

    st.divider()

    st.markdown('<div class="sidebar-title">User Profile</div>', unsafe_allow_html=True)
    st.write("User")

# ── MAIN HEADER ────────────────────────────────────────────
st.markdown('<div class="main-title">AI Content Authenticity System</div>', unsafe_allow_html=True)

# ── MODE SELECT ────────────────────────────────────────────
mode = st.radio("", ["Text", "Image"], horizontal=True)

# ── INPUT BOX (CENTER STYLE) ───────────────────────────────
st.markdown('<div class="input-container">', unsafe_allow_html=True)

result = None

col1, col2 = st.columns([6,1])

with col1:
    if mode == "Text":
        user_text = st.text_area(
            "",
            height=120,
            placeholder="Upload text or paste content..."
        )

    else:
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

with col2:
    analyze = st.button("→", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── PROCESSING ─────────────────────────────────────────────
if analyze:

    # TEXT
    if mode == "Text":
        if not user_text.strip():
            st.warning("Enter text first")
        else:
            with st.spinner("Processing..."):
                tokenizer, text_model = get_text_model()
                result = ensemble_predict(user_text, tokenizer, text_model)

    # IMAGE
    else:
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

            with st.spinner("Processing image..."):
                image_model = get_image_model()
                result = predict_image(image, image_model)
        else:
            st.warning("Upload image first")

# ── RESULT DISPLAY ─────────────────────────────────────────
if result:
    box_class = "ai" if result["label"] == "AI Generated" else "human"

    st.markdown(f'<div class="result-box {box_class}">', unsafe_allow_html=True)

    st.markdown(f"<h3>{result['label']}</h3>", unsafe_allow_html=True)
    st.markdown(f"<div class='score'>{result['score']}%</div>", unsafe_allow_html=True)
    st.write("AI Likelihood")

    col1, col2, col3 = st.columns(3)

    col1.metric("Confidence", f"{result['confidence']}%")

    if "roberta_ai" in result:
        col2.metric("RoBERTa", f"{result['roberta_ai']*100:.0f}%")
        col3.metric("Heuristic", f"{result['heuristic_ai']:.0f}%")
    else:
        col2.metric("AI Score", f"{result['score']}%")
        col3.metric("Real Probability", f"{round(result['real_probability']*100,2)}%")

    st.progress(result['ai_probability'])

    st.markdown('</div>', unsafe_allow_html=True)
