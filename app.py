import streamlit as st
from PIL import Image
import sys
import os

# ── Path setup ─────────────────────────────────────────────
sys.path.append(os.path.dirname(__file__))
from Pipeline.image_pipeline import load_image_model, predict_image
from Pipeline.text_pipeline import load_text_model, ensemble_predict

# ── Page Config ────────────────────────────────────────────
st.set_page_config(page_title="AI Detector", layout="wide")

# ── MODERN UI ──────────────────────────────────────────────
st.markdown("""
<style>

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

/* HEADER */
.header-box {
    padding: 30px;
    border-radius: 16px;
    background: linear-gradient(135deg, #1E293B, #334155);
    color: white;
    margin-bottom: 25px;
}

.header-title {
    font-size: 28px;
    font-weight: 600;
}

.header-sub {
    color: #CBD5F5;
    margin-top: 5px;
}

/* MODE BUTTONS */
.mode-btn {
    border-radius: 10px;
    padding: 10px 18px;
    border: 1px solid #E2E8F0;
    background: white;
    cursor: pointer;
    text-align: center;
    font-weight: 500;
}

.mode-active {
    background: #2563EB;
    color: white;
    border: none;
}

/* CARD */
.card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #E2E8F0;
    margin-top: 10px;
}

/* RESULT */
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

.big-score {
    font-size: 34px;
    font-weight: 600;
}

/* METRICS */
.metric {
    background: #F8FAFC;
    padding: 12px;
    border-radius: 8px;
    border: 1px solid #E2E8F0;
    text-align: center;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ── MODE STATE ─────────────────────────────────────────────
if "mode" not in st.session_state:
    st.session_state.mode = "Text"

# ── HEADER ─────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <div class="header-title">AI Content Authenticity Detector</div>
    <div class="header-sub">Detect whether content is AI-generated or human-written</div>
</div>
""", unsafe_allow_html=True)

# ── MODE SWITCH BUTTONS ────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    if st.button("Text Analysis", use_container_width=True):
        st.session_state.mode = "Text"

with col2:
    if st.button("Image Analysis", use_container_width=True):
        st.session_state.mode = "Image"

mode = st.session_state.mode

# ───────────────────────── TEXT MODE ───────────────────────
if mode == "Text":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    user_text = st.text_area(
        "Enter Text",
        height=180,
        placeholder="Paste your content here..."
    )

    if st.button("Analyze", use_container_width=True):

        if not user_text.strip():
            st.warning("Please enter text")
        else:
            with st.spinner("Analyzing..."):
                tokenizer, text_model = load_text_model("model/text_model")
                result = ensemble_predict(user_text, tokenizer, text_model)

            # UNCERTAIN LOGIC
            label = result["label"]
            if result["confidence"] < 60:
                label = "UNCERTAIN"
                box_class = "result-uncertain"
            else:
                box_class = "result-ai" if label == "AI Generated" else "result-human"

            st.markdown(f"""
            <div class="{box_class}">
                <b>{label}</b>
                <div class="big-score">{result['score']}%</div>
                <div>AI Likelihood</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)

            col1.markdown(f"<div class='metric'><b>Confidence</b><br>{result['confidence']}%</div>", unsafe_allow_html=True)
            col2.markdown(f"<div class='metric'><b>RoBERTa</b><br>{result['roberta_ai']*100:.0f}%</div>", unsafe_allow_html=True)
            col3.markdown(f"<div class='metric'><b>Heuristic</b><br>{result['heuristic_ai']:.0f}%</div>", unsafe_allow_html=True)
            col4.markdown(f"<div class='metric'><b>Prediction</b><br>{label}</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(result['ai_probability'])

    st.markdown('</div>', unsafe_allow_html=True)


# ───────────────────────── IMAGE MODE ──────────────────────
if mode == "Image":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, width=260)

        with col2:
            with st.spinner("Analyzing image..."):
                image_model = load_image_model("model/image_model/image_model.pth")
                result = predict_image(image, image_model)

            # UNCERTAIN LOGIC
            label = result["label"]
            if result["confidence"] < 60:
                label = "UNCERTAIN"
                box_class = "result-uncertain"
            else:
                box_class = "result-ai" if label == "AI Generated" else "result-human"

            st.markdown(f"""
            <div class="{box_class}">
                <b>{label}</b>
                <div class="big-score">{result['score']}%</div>
                <div>AI Likelihood</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            colA, colB, colC = st.columns(3)

            colA.markdown(f"<div class='metric'><b>Confidence</b><br>{result['confidence']}%</div>", unsafe_allow_html=True)
            colB.markdown(f"<div class='metric'><b>AI Score</b><br>{result['score']}%</div>", unsafe_allow_html=True)
            colC.markdown(f"<div class='metric'><b>Real Probability</b><br>{round(result['real_probability']*100,2)}%</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(result['ai_probability'])

    st.markdown('</div>', unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("AI Detection System • Final Project Submission")
