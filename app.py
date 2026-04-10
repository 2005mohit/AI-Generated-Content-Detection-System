import streamlit as st
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Pipeline.text_pipeline import ensemble_predict
from Pipeline.image_pipeline import load_image_model, predict_image

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Content Authenticity Detector",
    page_icon="🤖",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  /* Hero header */
  .hero-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    text-align: center;
  }
  .hero-header h1 {
    color: #ffffff;
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
  }
  .hero-header p {
    color: #a0aec0;
    font-size: 1rem;
    margin-bottom: 1rem;
  }
  .badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    justify-content: center;
  }
  .badge {
    background: rgba(255,255,255,0.1);
    color: #e2e8f0;
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 999px;
    padding: 0.25rem 0.85rem;
    font-size: 0.78rem;
    font-weight: 500;
  }

  /* Result boxes */
  .result-ai {
    background: linear-gradient(135deg, #fff5f5, #fed7d7);
    border: 2px solid #fc8181;
    border-radius: 14px;
    padding: 1.4rem;
    text-align: center;
    margin: 1rem 0;
  }
  .result-human {
    background: linear-gradient(135deg, #f0fff4, #c6f6d5);
    border: 2px solid #68d391;
    border-radius: 14px;
    padding: 1.4rem;
    text-align: center;
    margin: 1rem 0;
  }
  .result-uncertain {
    background: linear-gradient(135deg, #fffff0, #fefcbf);
    border: 2px solid #f6e05e;
    border-radius: 14px;
    padding: 1.4rem;
    text-align: center;
    margin: 1rem 0;
  }
  .result-title {
    font-size: 1.5rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
  }
  .result-sub {
    font-size: 0.85rem;
    color: #555;
  }

  /* Metric cards */
  .metric-row {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-top: 0.8rem;
  }
  .metric-card {
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    flex: 1;
    min-width: 100px;
    text-align: center;
  }
  .metric-value {
    font-size: 1.3rem;
    font-weight: 800;
    color: #2d3748;
  }
  .metric-label {
    font-size: 0.72rem;
    color: #718096;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.1rem;
  }

  /* Section labels */
  .section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #718096;
    margin-bottom: 0.4rem;
  }

  /* Warning box */
  .warning-box {
    background: #fffbeb;
    border: 1px solid #f6e05e;
    border-radius: 10px;
    padding: 0.7rem 1rem;
    font-size: 0.85rem;
    color: #744210;
    margin-bottom: 0.8rem;
  }

  /* Footer */
  .footer {
    text-align: center;
    color: #a0aec0;
    font-size: 0.78rem;
    margin-top: 3rem;
    padding: 1rem;
    border-top: 1px solid #e2e8f0;
  }

  /* Hide default Streamlit padding */
  .block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Hero Header ───────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <h1>🤖 AI Content Authenticity Detector</h1>
  <p>Detect whether text or image is AI-Generated or Human/Real</p>
  <div class="badge-row">
    <span class="badge">🧠 RoBERTa Model</span>
    <span class="badge">👁️ EfficientNet B0</span>
    <span class="badge">⚡ Real-Time Inference</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Load image model (cached) ─────────────────────────────────
@st.cache_resource
def get_image_model():
    return load_image_model()

image_model = get_image_model()

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📝 Text Analysis", "🖼️ Image Analysis"])

# ════════════════════════════════════════════════════════════════
# TAB 1 — TEXT DETECTION
# ════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-label">Enter text to analyze</p>', unsafe_allow_html=True)
    st.markdown("**Uses RoBERTa (chatgpt-detector-roberta) model**")

    user_text = st.text_area(
        label="Input Text",
        placeholder="Paste or type your text here...",
        height=200,
        label_visibility="collapsed"
    )

    analyze_text = st.button("🔍 Analyze Text", use_container_width=True)

    if analyze_text:
        if not user_text or not user_text.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            word_count = len(user_text.strip().split())

            if word_count < 10:
                st.markdown("""
                <div class="warning-box">
                  ⚠️ <strong>Short Text Warning</strong> — Less than 10 words detected.
                  Results may not be reliable. Try providing more content.
                </div>
                """, unsafe_allow_html=True)

            with st.spinner("Analyzing text..."):
                result = ensemble_predict(user_text)

            ai_score    = result.get("ai_score", 0)
            confidence  = result.get("confidence", 0)
            label       = result.get("label", "Unknown")
            roberta_pct = result.get("roberta_score", 0)
            heuristic_pct = result.get("heuristic_score", 0)

            # ── Result Box ────────────────────────────────────
            if confidence < 0.60:
                st.markdown(f"""
                <div class="result-uncertain">
                  <div class="result-title">⚠️ UNCERTAIN</div>
                  <div class="result-sub">Low Confidence — Result may not be reliable<br>
                  Try providing more content for better accuracy</div>
                </div>
                """, unsafe_allow_html=True)
            elif ai_score > 0.60:
                st.markdown(f"""
                <div class="result-ai">
                  <div class="result-title">🤖 AI Generated</div>
                  <div class="result-sub">RoBERTa: {label} | Heuristic: AI Generated</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-human">
                  <div class="result-title">✅ Human Written</div>
                  <div class="result-sub">RoBERTa: {label} | Heuristic: Human Generated</div>
                </div>
                """, unsafe_allow_html=True)

            # ── Metric Cards ──────────────────────────────────
            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-card">
                <div class="metric-value">{ai_score*100:.1f}%</div>
                <div class="metric-label">AI Score</div>
              </div>
              <div class="metric-card">
                <div class="metric-value">{confidence*100:.1f}%</div>
                <div class="metric-label">Confidence</div>
              </div>
              <div class="metric-card">
                <div class="metric-value">{roberta_pct*100:.0f}%</div>
                <div class="metric-label">RoBERTa</div>
              </div>
              <div class="metric-card">
                <div class="metric-value">{heuristic_pct*100:.0f}%</div>
                <div class="metric-label">Heuristic</div>
              </div>
              <div class="metric-card">
                <div class="metric-value">{word_count}</div>
                <div class="metric-label">Words</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Progress Bar ──────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"**AI Likelihood: {ai_score*100:.2f}%**")
            st.progress(float(ai_score))

# ════════════════════════════════════════════════════════════════
# TAB 2 — IMAGE DETECTION
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-label">Upload an image to analyze</p>', unsafe_allow_html=True)
    st.markdown("**Uses EfficientNet model**")

    uploaded_file = st.file_uploader(
        label="Upload Image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        # ── Display image using st.image with fixed width ─────
        st.image(image, width=260)

        analyze_img = st.button("🔍 Analyze Image", use_container_width=True)

        if analyze_img:
            with st.spinner("Analyzing image..."):
                img_result = predict_image(image_model, image)

            ai_score_img   = img_result.get("ai_score", 0)
            confidence_img = img_result.get("confidence", 0)
            real_prob      = img_result.get("real_prob", 0)
            label_img      = img_result.get("label", "Unknown")

            # ── Result Box ────────────────────────────────────
            if confidence_img < 0.60:
                st.markdown(f"""
                <div class="result-uncertain">
                  <div class="result-title">⚠️ UNCERTAIN</div>
                  <div class="result-sub">Low Confidence — Result may not be reliable</div>
                </div>
                """, unsafe_allow_html=True)
            elif ai_score_img > 0.50:
                st.markdown(f"""
                <div class="result-ai">
                  <div class="result-title">🤖 AI Generated</div>
                  <div class="result-sub">EfficientNet confidence: {confidence_img*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-human">
                  <div class="result-title">✅ Real / Human</div>
                  <div class="result-sub">EfficientNet confidence: {confidence_img*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            # ── Metric Cards ──────────────────────────────────
            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-card">
                <div class="metric-value">{ai_score_img*100:.1f}%</div>
                <div class="metric-label">AI Score</div>
              </div>
              <div class="metric-card">
                <div class="metric-value">{confidence_img*100:.1f}%</div>
                <div class="metric-label">Confidence</div>
              </div>
              <div class="metric-card">
                <div class="metric-value">{real_prob*100:.1f}%</div>
                <div class="metric-label">Real Prob</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Progress Bar ──────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"**AI Likelihood: {ai_score_img*100:.2f}%**")
            st.progress(float(ai_score_img))

    else:
        st.info("Upload a JPG, JPEG, PNG or WEBP image to get started.")

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  AI-Generated Content Detection System &nbsp;|&nbsp; 2026
</div>
""", unsafe_allow_html=True)
