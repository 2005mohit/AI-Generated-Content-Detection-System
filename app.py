import streamlit as st
from PIL import Image
import sys
import os

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Content Authenticity Detector",
    page_icon="🤖",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide default streamlit header */
#MainMenu, footer, header {visibility: hidden;}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    padding: 2rem 2.5rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 0.95rem;
    color: #a0aec0;
    margin: 0 0 1.2rem 0;
}
.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
}
.badge {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.15);
    color: #e2e8f0;
    font-size: 0.78rem;
    padding: 4px 12px;
    border-radius: 999px;
    font-weight: 500;
}

/* Result boxes */
.result-ai {
    background: linear-gradient(135deg, #2d0a0a, #450e0e);
    border: 1px solid #ff4444;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    color: #ff6b6b;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 1rem 0;
    box-shadow: 0 0 20px rgba(255,68,68,0.2);
}
.result-human {
    background: linear-gradient(135deg, #0a2d0a, #0e4512);
    border: 1px solid #22c55e;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    color: #4ade80;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 1rem 0;
    box-shadow: 0 0 20px rgba(34,197,94,0.2);
}
.result-uncertain {
    background: linear-gradient(135deg, #2d2200, #4a3800);
    border: 1px solid #f59e0b;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    color: #fbbf24;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 1rem 0;
    box-shadow: 0 0 20px rgba(245,158,11,0.2);
}
.result-sub {
    font-size: 0.82rem;
    margin-top: 4px;
    opacity: 0.75;
    font-weight: 400;
}

/* Metric card */
.metric-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin: 0.8rem 0;
}
.metric-card {
    background: #1e1e2e;
    border: 1px solid #2e2e3e;
    border-radius: 10px;
    padding: 0.7rem 1.1rem;
    flex: 1;
    min-width: 100px;
    text-align: center;
}
.metric-label {
    font-size: 0.72rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: #e2e8f0;
}

/* Progress bar custom */
.progress-wrap {
    background: #1e1e2e;
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
    margin: 0.5rem 0 1.2rem 0;
}
.progress-fill-ai   { height: 8px; background: linear-gradient(90deg,#ef4444,#ff6b6b); border-radius:999px; transition: width 0.6s ease; }
.progress-fill-human{ height: 8px; background: linear-gradient(90deg,#22c55e,#4ade80); border-radius:999px; transition: width 0.6s ease; }
.progress-fill-unc  { height: 8px; background: linear-gradient(90deg,#f59e0b,#fbbf24); border-radius:999px; transition: width 0.6s ease; }

/* Hint text */
.hint-text {
    font-size: 0.8rem;
    color: #6b7280;
    margin-top: 0.3rem;
}

/* Image preview container */
.img-container {
    background: #1a1a2a;
    border: 1px solid #2e2e3e;
    border-radius: 10px;
    padding: 10px;
    margin: 0.8rem 0;
    display: inline-block;
}

/* Footer */
.footer {
    text-align: center;
    color: #4b5563;
    font-size: 0.78rem;
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid #1e1e2e;
}

/* Divider labels */
.section-label {
    font-size: 0.72rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}

/* Warning box */
.warn-box {
    background: #2d2200;
    border: 1px solid #92400e;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    color: #fbbf24;
    font-size: 0.85rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Pipeline imports (safe fallback for demo) ─────────────────────────────────
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), "Pipeline"))
    from text_pipeline import load_text_model, ensemble_predict
    from image_pipeline import load_image_model, predict_image
    TEXT_MODEL  = load_text_model("model/text_model")
    IMAGE_MODEL = load_image_model("model/image_model/image_model.pth")
    MODELS_LOADED = True
except Exception as e:
    MODELS_LOADED = False
    MODEL_ERROR = str(e)

# ── Hero Banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <p class="hero-title">🤖 AI Content Authenticity Detector</p>
  <p class="hero-subtitle">Detect whether text or image is AI-Generated or Human / Real</p>
  <div class="badge-row">
    <span class="badge">📝 RoBERTa Ensemble</span>
    <span class="badge">🖼️ EfficientNet B0</span>
    <span class="badge">✅ 90% Text Accuracy</span>
    <span class="badge">⚡ Real-time &lt;3s</span>
    <span class="badge">🎓 KR Mangalam University</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not MODELS_LOADED:
    st.warning(f"⚠️ Models not loaded — running in demo mode. Error: `{MODEL_ERROR}`", icon="⚠️")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📝  Text Analysis", "🖼️  Image Analysis"])

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — TEXT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-label">Uses RoBERTa (chatgpt-detector-roberta) + Heuristic Ensemble</p>', unsafe_allow_html=True)

    user_text = st.text_area(
        "Enter text to analyze:",
        placeholder="Paste or type any text here — AI-generated or human-written...",
        height=180,
        label_visibility="visible"
    )

    analyze_text = st.button("🔍 Analyze Text", use_container_width=True, key="btn_text")

    if analyze_text:
        if not user_text or len(user_text.strip()) == 0:
            st.error("⚠️ Please enter some text first.")
        else:
            word_count = len(user_text.strip().split())

            if word_count < 10:
                st.markdown('<div class="warn-box">⚠️ Very short text detected (&lt;10 words). Results may be unreliable — try providing more content.</div>', unsafe_allow_html=True)

            with st.spinner("Analyzing text..."):
                if MODELS_LOADED:
                    result = ensemble_predict(user_text, TEXT_MODEL)
                    ai_score    = result["ai_score"]
                    confidence  = result["confidence"]
                    roberta_pct = result["roberta_score"] * 100
                    heuristic   = result["heuristic_score"] * 100
                else:
                    # Demo fallback values
                    import random
                    ai_score    = round(random.uniform(0.4, 0.99), 4)
                    confidence  = round(random.uniform(0.5, 0.95), 4)
                    roberta_pct = round(ai_score * 100 * random.uniform(0.85, 1.0), 1)
                    heuristic   = round(ai_score * 100 * random.uniform(0.85, 1.0), 1)

            ai_pct = round(ai_score * 100, 2)
            conf_pct = round(confidence * 100, 1)

            # ── Result Label ──────────────────────────────────────────
            if conf_pct < 60:
                label = "uncertain"
                st.markdown(f'<div class="result-uncertain">⚠️ UNCERTAIN<div class="result-sub">Low Confidence — Result may not be reliable. Try providing more content.</div></div>', unsafe_allow_html=True)
            elif ai_pct > 60:
                label = "ai"
                st.markdown(f'<div class="result-ai">🤖 AI GENERATED<div class="result-sub">Content shows strong AI-generated patterns</div></div>', unsafe_allow_html=True)
            else:
                label = "human"
                st.markdown(f'<div class="result-human">✅ HUMAN WRITTEN<div class="result-sub">Content appears to be human-authored</div></div>', unsafe_allow_html=True)

            # ── Metric Cards ──────────────────────────────────────────
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-card">
                    <div class="metric-label">AI Score</div>
                    <div class="metric-value">{ai_pct}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{conf_pct}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">RoBERTa</div>
                    <div class="metric-value">{round(roberta_pct, 1)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Heuristic</div>
                    <div class="metric-value">{round(heuristic, 1)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Words</div>
                    <div class="metric-value">{word_count}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Progress Bar ──────────────────────────────────────────
            fill_class = "progress-fill-ai" if label == "ai" else ("progress-fill-human" if label == "human" else "progress-fill-unc")
            st.markdown(f'<p class="hint-text">AI Likelihood: {ai_pct}%</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="progress-wrap"><div class="{fill_class}" style="width:{ai_pct}%"></div></div>', unsafe_allow_html=True)

            # ── Detail note ───────────────────────────────────────────
            r_label = "AI Generated" if roberta_pct > 60 else "Human"
            h_label = "AI Generated" if heuristic > 60 else "Human"
            st.markdown(f'<p class="hint-text">RoBERTa: {r_label} &nbsp;|&nbsp; Heuristic: {h_label}</p>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — IMAGE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-label">Uses EfficientNet B0 — fine-tuned on 140k+ face & COCOAI dataset</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload an image to analyze:",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="visible"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        # ── Two-column layout: image left, analyze button right ───────
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown('<p class="section-label">Uploaded Image</p>', unsafe_allow_html=True)
            st.image(image, width=260)   # ← your requested logic

        with col2:
            st.markdown('<p class="section-label">Analysis</p>', unsafe_allow_html=True)
            analyze_img = st.button("🔍 Analyze Image", use_container_width=True, key="btn_img")

            if analyze_img:
                with st.spinner("Running EfficientNet inference..."):
                    if MODELS_LOADED:
                        result = predict_image(image, IMAGE_MODEL)
                        ai_prob   = result["ai_prob"]
                        real_prob = result["real_prob"]
                        conf      = result["confidence"]
                    else:
                        import random
                        ai_prob   = round(random.uniform(0.3, 0.99), 4)
                        real_prob = round(1 - ai_prob, 4)
                        conf      = round(max(ai_prob, real_prob), 4)

                ai_pct   = round(ai_prob * 100, 2)
                real_pct = round(real_prob * 100, 2)
                conf_pct = round(conf * 100, 2)

                # ── Result Label ──────────────────────────────────────
                if conf_pct < 60:
                    label = "uncertain"
                    st.markdown('<div class="result-uncertain">⚠️ UNCERTAIN<div class="result-sub">Low confidence — try a clearer image</div></div>', unsafe_allow_html=True)
                elif ai_pct > 60:
                    label = "ai"
                    st.markdown(f'<div class="result-ai">🤖 AI GENERATED<div class="result-sub">Synthetic image patterns detected</div></div>', unsafe_allow_html=True)
                else:
                    label = "human"
                    st.markdown(f'<div class="result-human">✅ REAL / HUMAN<div class="result-sub">Image appears authentic</div></div>', unsafe_allow_html=True)

                # ── Metric Cards ──────────────────────────────────────
                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-card">
                        <div class="metric-label">AI Score</div>
                        <div class="metric-value">{ai_pct}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value">{conf_pct}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Real Prob</div>
                        <div class="metric-value">{real_pct}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Progress Bar ──────────────────────────────────────
                fill_class = "progress-fill-ai" if label == "ai" else ("progress-fill-human" if label == "human" else "progress-fill-unc")
                st.markdown(f'<p class="hint-text">AI Likelihood: {ai_pct}%</p>', unsafe_allow_html=True)
                st.markdown(f'<div class="progress-wrap"><div class="{fill_class}" style="width:{ai_pct}%"></div></div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🎓 K.R. Mangalam University &nbsp;|&nbsp; AI-Generated Content Detection System &nbsp;|&nbsp; 2026<br>
    Built with Streamlit · RoBERTa · EfficientNet B0
</div>
""", unsafe_allow_html=True)
