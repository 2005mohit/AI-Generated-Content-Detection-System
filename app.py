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
    page_title="AI Content detaction system",
    page_icon="🤖",
    layout="wide"
)

# ── CUSTOM DARK UI (ChatGPT Style) ─────────────────────────
st.markdown("""
<style>
    body {
        background-color: #0B0F14;
        color: white;
    }

    .main-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sub-text {
        color: #9CA3AF;
        margin-bottom: 25px;
    }

    .card {
        background: #111827;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0px 0px 20px rgba(0,0,0,0.4);
    }

    .upload-box {
        border: 2px dashed #374151;
        padding: 40px;
        border-radius: 16px;
        text-align: center;
        transition: 0.3s;
    }

    .upload-box:hover {
        border-color: #22c55e;
        background: #0f172a;
    }

    .result-ai {
        background: #1f2937;
        border-left: 5px solid #ef4444;
        padding: 20px;
        border-radius: 12px;
    }

    .result-human {
        background: #1f2937;
        border-left: 5px solid #22c55e;
        padding: 20px;
        border-radius: 12px;
    }

    .metric-card {
        background: #111827;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
    }

    .big-score {
        font-size: 40px;
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

# ── HEADER (MODEL NAME LIKE CHATGPT UI) ────────────────────
st.markdown('<div class="main-title">🤖 AI Content Authenticity Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Analyze text & images with AI detection models (RoBERTa + EfficientNet)</div>', unsafe_allow_html=True)

# ── MODE SELECT (CHATGPT STYLE) ────────────────────────────
mode = st.radio(
    "",
    ["📝 Text Analysis", "🖼️ Image Analysis"],
    horizontal=True
)

st.divider()

# ───────────────────────── TEXT MODE ───────────────────────
if "Text" in mode:

    st.markdown("### 🧠 Text Detection (RoBERTa Model)")

    user_text = st.text_area(
        "",
        height=200,
        placeholder="Paste your content here..."
    )

    analyze = st.button("🚀 Analyze Text", use_container_width=True)

    if analyze:
        if not user_text.strip():
            st.warning("⚠️ Please enter text")
        else:
            with st.spinner("Analyzing..."):
                tokenizer, text_model = get_text_model()
                result = ensemble_predict(user_text, tokenizer, text_model)

            box_class = "result-ai" if result["label"] == "AI Generated" else "result-human"

            # RESULT CARD
            st.markdown(f"""
            <div class="{box_class}">
                <h2>{result['label']}</h2>
                <div class="big-score">{result['score']}%</div>
                <p>AI Likelihood</p>
            </div>
            """, unsafe_allow_html=True)

            # METRICS
            col1, col2, col3, col4 = st.columns(4)

            col1.markdown(f"<div class='metric-card'><b>Confidence</b><br>{result['confidence']}%</div>", unsafe_allow_html=True)
            col2.markdown(f"<div class='metric-card'><b>RoBERTa</b><br>{result['roberta_ai']*100:.0f}%</div>", unsafe_allow_html=True)
            col3.markdown(f"<div class='metric-card'><b>Heuristic</b><br>{result['heuristic_ai']:.0f}%</div>", unsafe_allow_html=True)
            col4.markdown(f"<div class='metric-card'><b>Label</b><br>{result['label']}</div>", unsafe_allow_html=True)

            st.progress(result['ai_probability'], text=f"AI Likelihood: {result['score']}%")


# ───────────────────────── IMAGE MODE ──────────────────────
if "Image" in mode:

    st.markdown("### 🖼️ Image Detection (EfficientNet Model)")

    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns([1,1])

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            with st.spinner("Analyzing image automatically..."):
                image_model = get_image_model()
                result = predict_image(image, image_model)

            box_class = "result-ai" if result["label"] == "AI Generated" else "result-human"

            st.markdown(f"""
            <div class="{box_class}">
                <h2>{result['label']}</h2>
                <div class="big-score">{result['score']}%</div>
                <p>AI Likelihood</p>
            </div>
            """, unsafe_allow_html=True)

            # METRICS
            colA, colB, colC = st.columns(3)

            colA.markdown(f"<div class='metric-card'><b>Confidence</b><br>{result['confidence']}%</div>", unsafe_allow_html=True)
            colB.markdown(f"<div class='metric-card'><b>AI Score</b><br>{result['score']}%</div>", unsafe_allow_html=True)
            colC.markdown(f"<div class='metric-card'><b>Real Prob</b><br>{round(result['real_probability']*100,2)}%</div>", unsafe_allow_html=True)

            st.progress(result['ai_probability'], text=f"AI Likelihood: {result['score']}%")

# ── FOOTER ────────────────────────────────────────────────
st.divider()
st.caption("🚀 Built by Mohit | AI Detection System")
