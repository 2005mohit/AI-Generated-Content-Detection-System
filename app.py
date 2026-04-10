import streamlit as st
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Pipeline.text_pipeline import load_text_model, ensemble_predict
from Pipeline.image_pipeline import load_image_model, predict_image

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Content Authenticity Detector",
    page_icon="🤖",
    layout="wide"
)

# ── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container {max-width: 1100px; padding-top: 1.5rem;}

.hero {
    background: linear-gradient(135deg, #1e293b, #334155);
    padding: 28px;
    border-radius: 14px;
    text-align: center;
    margin-bottom: 25px;
}
.hero h1 {color:white; margin-bottom:5px;}
.hero p {color:#cbd5e1; font-size:14px;}

.result-ai {
    background:#fff1f2;
    border-left:4px solid #ef4444;
    padding:16px;
    border-radius:10px;
}
.result-human {
    background:#f0fdf4;
    border-left:4px solid #22c55e;
    padding:16px;
    border-radius:10px;
}
.result-uncertain {
    background:#fffbeb;
    border-left:4px solid #f59e0b;
    padding:16px;
    border-radius:10px;
}
.big {font-size:32px; font-weight:600;}

.metric {
    background:#f8fafc;
    padding:10px;
    border-radius:8px;
    border:1px solid #e2e8f0;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ── HEADER ──────────────────────────────────────────────────
st.markdown("""
<div class="hero">
<h1>AI Content Authenticity Detector</h1>
<p>Detect whether text or images are AI-generated or human-created</p>
</div>
""", unsafe_allow_html=True)

# ── LOAD MODELS ─────────────────────────────────────────────
@st.cache_resource
def get_text_model():
    return load_text_model("model/text_model")

@st.cache_resource
def get_image_model():
    return load_image_model()

tokenizer, text_model = get_text_model()
image_model = get_image_model()

# ── TABS ────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📝 Text Analysis", "🖼️ Image Analysis"])

# ════════════════════════════════════════════════════════════
# TEXT ANALYSIS
# ════════════════════════════════════════════════════════════
with tab1:

    user_text = st.text_area(
        "Enter text",
        height=180,
        placeholder="Paste your text here..."
    )

    if st.button("Analyze Text", use_container_width=True):

        if not user_text.strip():
            st.warning("Please enter text")
        else:
            with st.spinner("Analyzing..."):
                result = ensemble_predict(user_text, tokenizer, text_model)

            ai_score = result.get("ai_score", 0)
            confidence = result.get("confidence", 0)
            label = result.get("label", "")

            # ── RESULT ──
            if confidence < 0.60:
                st.markdown(f"""
                <div class="result-uncertain">
                <b>UNCERTAIN</b>
                <div class="big">{ai_score*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            elif ai_score > 0.5:
                st.markdown(f"""
                <div class="result-ai">
                <b>AI Generated</b>
                <div class="big">{ai_score*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div class="result-human">
                <b>Human Written</b>
                <div class="big">{ai_score*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            # ── METRICS ──
            col1, col2, col3 = st.columns(3)
            col1.markdown(f"<div class='metric'>Confidence<br><b>{confidence*100:.1f}%</b></div>", unsafe_allow_html=True)
            col2.markdown(f"<div class='metric'>AI Score<br><b>{ai_score*100:.1f}%</b></div>", unsafe_allow_html=True)
            col3.markdown(f"<div class='metric'>Prediction<br><b>{label}</b></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(float(ai_score))

# ════════════════════════════════════════════════════════════
# IMAGE ANALYSIS
# ════════════════════════════════════════════════════════════
with tab2:

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1.2])

        # ── IMAGE ──
        with col1:
            st.image(image, width=280)

        # ── RESULT SIDE ──
        with col2:

            if st.button("Analyze Image", use_container_width=True):

                with st.spinner("Analyzing image..."):
                    result = predict_image(image, image_model)  # FIXED

                ai_score = result.get("ai_score", 0)
                confidence = result.get("confidence", 0)
                real_prob = result.get("real_prob", 0)

                # ── RESULT ──
                if confidence < 0.60:
                    st.markdown("""
                    <div class="result-uncertain">
                    <b>UNCERTAIN</b>
                    </div>
                    """, unsafe_allow_html=True)

                elif ai_score > 0.5:
                    st.markdown(f"""
                    <div class="result-ai">
                    <b>AI Generated</b>
                    <div class="big">{ai_score*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.markdown(f"""
                    <div class="result-human">
                    <b>Real Image</b>
                    <div class="big">{ai_score*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── METRICS ──
                colA, colB, colC = st.columns(3)
                colA.markdown(f"<div class='metric'>Confidence<br><b>{confidence*100:.1f}%</b></div>", unsafe_allow_html=True)
                colB.markdown(f"<div class='metric'>AI Score<br><b>{ai_score*100:.1f}%</b></div>", unsafe_allow_html=True)
                colC.markdown(f"<div class='metric'>Real Prob<br><b>{real_prob*100:.1f}%</b></div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.progress(float(ai_score))

# ── FOOTER ─────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("AI Content Authenticity Detection System • Final Submission")
