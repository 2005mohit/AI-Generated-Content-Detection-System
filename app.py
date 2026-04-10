import streamlit as st
import torch
import torchvision.transforms as transforms
from transformers import pipeline
from PIL import Image
import re
import math
from collections import Counter

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Content Authenticity Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Remove default top padding / empty block */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
}
[data-testid="stAppViewBlockContainer"] > div:first-child {
    margin-top: 0 !important;
}

/* Global font */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* Hero header */
.hero-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem 1.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255,255,255,0.08);
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0.35rem;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    color: rgba(255,255,255,0.6);
    font-size: 0.95rem;
    margin-bottom: 1rem;
}
.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 999px;
    padding: 0.25rem 0.75rem;
    font-size: 0.78rem;
    color: rgba(255,255,255,0.85);
    font-weight: 500;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: #f8f9fa;
    padding: 0.4rem;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0.5rem 1.5rem;
    color: #6b7280;
}
.stTabs [aria-selected="true"] {
    background-color: #ffffff !important;
    color: #1a1a2e !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

/* Section label */
.section-label {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9ca3af;
    margin-bottom: 0.4rem;
}
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 0.25rem;
}
.section-hint {
    font-size: 0.82rem;
    color: #9ca3af;
    margin-bottom: 1.2rem;
}

/* Result boxes */
.result-ai {
    background: #fff1f2;
    border: 2px solid #f87171;
    border-radius: 14px;
    padding: 1.5rem;
    margin-top: 1rem;
}
.result-human {
    background: #f0fdf4;
    border: 2px solid #4ade80;
    border-radius: 14px;
    padding: 1.5rem;
    margin-top: 1rem;
}
.result-uncertain {
    background: #fffbeb;
    border: 2px solid #fbbf24;
    border-radius: 14px;
    padding: 1.5rem;
    margin-top: 1rem;
}
.result-tag {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}
.result-tag-ai    { color: #dc2626; }
.result-tag-human { color: #16a34a; }
.result-tag-uncertain { color: #d97706; }

.result-score {
    font-size: 2.8rem;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.result-score-ai       { color: #dc2626; }
.result-score-human    { color: #16a34a; }
.result-score-uncertain{ color: #d97706; }

.result-label {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 0.5rem;
}
.result-note {
    font-size: 0.82rem;
    color: #6b7280;
    margin-top: 0.5rem;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 0.75rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.metric-card {
    flex: 1;
    min-width: 100px;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    text-align: center;
}
.metric-value {
    font-size: 1.4rem;
    font-weight: 800;
    color: #1f2937;
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #9ca3af;
    margin-top: 0.15rem;
}

/* Image display */
.img-wrapper {
    width: 100%;
    overflow: hidden;
    border-radius: 12px;
    background: #f3f4f6;
    max-height: 320px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.img-wrapper img {
    width: 100%;
    height: 320px;
    object-fit: cover;
    border-radius: 12px;
}

/* Warning box */
.warning-box {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-size: 0.84rem;
    color: #92400e;
    margin-top: 0.75rem;
}

/* Footer */
.footer {
    text-align: center;
    padding: 1.5rem 0 0;
    font-size: 0.78rem;
    color: #d1d5db;
    border-top: 1px solid #f3f4f6;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Hero Header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="hero-title">🔍 AI Content Authenticity Detector</div>
  <div class="hero-subtitle">Detect whether text or images are AI-Generated or Human/Real</div>
  <div class="badge-row">
    <span class="badge">🤖 RoBERTa Model</span>
    <span class="badge">🖼️ EfficientNet B0</span>
    <span class="badge">✅ 90% Text Accuracy</span>
    <span class="badge">⚡ Real-time • &lt;3s</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Model Loading ───────────────────────────────────────────────────────────
@st.cache_resource
def load_text_model():
    return pipeline("text-classification", model="roberta-base-openai-detector", top_k=None)


@st.cache_resource
def load_image_model():
    import torchvision.models as models
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    try:
        model.load_state_dict(torch.load("model/image_model/image_model.pth", map_location="cpu"))
        model.eval()
        return model, True
    except Exception:
        return model, False


# ─── Heuristic Detector ──────────────────────────────────────────────────────
def heuristic_score(text: str) -> float:
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        return 0.5
    words = text.lower().split()
    avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
    vocab_div = len(set(words)) / max(len(words), 1)
    bigrams = list(zip(words, words[1:]))
    bg_freq = Counter(bigrams)
    burstiness = (max(bg_freq.values()) if bg_freq else 1) / max(len(bigrams), 1)
    # AI text tends to: longer avg sentence, lower vocab diversity, higher bigram repeat
    score = 0.0
    if avg_len > 20: score += 0.3
    elif avg_len > 15: score += 0.15
    if vocab_div < 0.55: score += 0.3
    elif vocab_div < 0.65: score += 0.15
    if burstiness > 0.05: score += 0.2
    try:
        log_prob = sum(math.log(max(bg_freq.get(bg, 1) / max(len(bigrams), 1), 1e-6)) for bg in bigrams) / max(len(bigrams), 1)
        if log_prob > -4: score += 0.2
    except Exception:
        pass
    return min(score, 1.0)


# ─── Image Transform ─────────────────────────────────────────────────────────
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ─── Result Renderer ─────────────────────────────────────────────────────────
def render_result(label: str, confidence: float, ai_score: float, metrics: dict):
    if confidence < 60:
        box_cls = "result-uncertain"
        tag_cls = "result-tag-uncertain"
        score_cls = "result-score-uncertain"
        tag_text = "⚠️ UNCERTAIN"
        label_text = "Low Confidence — Result may not be reliable"
        note = "The model is not confident enough to make a definitive prediction. Try providing more content."
    elif label == "AI Generated":
        box_cls = "result-ai"
        tag_cls = "result-tag-ai"
        score_cls = "result-score-ai"
        tag_text = "🤖 AI GENERATED"
        label_text = "AI Generated Content Detected"
        note = "This content shows strong indicators of AI generation."
    else:
        box_cls = "result-human"
        tag_cls = "result-tag-human"
        score_cls = "result-score-human"
        tag_text = "✅ HUMAN WRITTEN"
        label_text = "Human-Created Content"
        note = "This content appears to be written or created by a human."

    metric_html = "".join([
        f'<div class="metric-card"><div class="metric-value">{v}</div><div class="metric-label">{k}</div></div>'
        for k, v in metrics.items()
    ])

    st.markdown(f"""
    <div class="{box_cls}">
      <div class="result-tag {tag_cls}">{tag_text}</div>
      <div class="result-score {score_cls}">{ai_score:.1f}%</div>
      <div class="result-label">{label_text}</div>
      <div class="metric-row">{metric_html}</div>
      <div class="result-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📝  Text Analysis", "🖼️  Image Analysis"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TEXT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-label">Text Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Text AI Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-hint">Uses RoBERTa (roberta-base-openai-detector) + Heuristic Ensemble</div>', unsafe_allow_html=True)

    user_text = st.text_area(
        "Enter text to analyze:",
        height=200,
        placeholder="Paste or type your text here (minimum ~40 words recommended for best accuracy)…",
        label_visibility="collapsed"
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        analyze_text = st.button("🔍 Analyze Text", use_container_width=True, type="primary")

    if analyze_text:
        if not user_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            words = user_text.split()
            if len(words) < 10:
                st.markdown('<div class="warning-box">⚠️ Text is very short (&lt;10 words). Results may be unreliable. Please provide more content.</div>', unsafe_allow_html=True)

            with st.spinner("Analyzing text…"):
                try:
                    clf = load_text_model()
                    results = clf(user_text[:512])
                    # Handle various output shapes
                    if isinstance(results[0], list):
                        scores_raw = {r["label"].upper(): r["score"] for r in results[0]}
                    else:
                        scores_raw = {results[0]["label"].upper(): results[0]["score"]}

                    roberta_ai = scores_raw.get("FAKE", scores_raw.get("AI", 0.5))
                    roberta_human = 1 - roberta_ai
                    heuristic = heuristic_score(user_text)

                    final_score = 0.40 * roberta_ai + 0.60 * heuristic
                    final_pct = final_score * 100
                    confidence = max(final_pct, 100 - final_pct)

                    label = "AI Generated" if final_score > 0.60 else "Human Written"
                    roberta_label = "AI Generated" if roberta_ai > 0.5 else "Human Written"
                    heuristic_label = "AI Generated" if heuristic > 0.5 else "Human Written"

                    render_result(label, confidence, final_pct, {
                        "AI Score": f"{final_pct:.1f}%",
                        "Confidence": f"{confidence:.1f}%",
                        "RoBERTa": f"{roberta_ai*100:.0f}%",
                        "Heuristic": f"{heuristic*100:.0f}%",
                        "Word Count": str(len(words))
                    })

                    st.markdown(f"""
                    <div style="margin-top:0.75rem; font-size:0.82rem; color:#6b7280;">
                      RoBERTa: <strong>{roberta_label}</strong> &nbsp;|&nbsp;
                      Heuristic: <strong>{heuristic_label}</strong> &nbsp;|&nbsp;
                      AI Likelihood: <strong>{final_pct:.2f}%</strong>
                    </div>
                    """, unsafe_allow_html=True)

                    st.progress(min(final_pct / 100, 1.0))

                except Exception as e:
                    st.error(f"Text analysis error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — IMAGE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-label">Image Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Image AI Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-hint">Uses EfficientNet B0 (fine-tuned) — Supports JPG, JPEG, PNG, WEBP</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        # Show image zoomed out / contained
        st.markdown('<div class="img-wrapper">', unsafe_allow_html=True)
        st.image(image, use_container_width=True, caption="Uploaded Image")
        st.markdown('</div>', unsafe_allow_html=True)

        col_btn2, _ = st.columns([1, 4])
        with col_btn2:
            analyze_img = st.button("🔍 Analyze Image", use_container_width=True, type="primary")

        if analyze_img:
            with st.spinner("Analyzing image…"):
                try:
                    model, loaded = load_image_model()

                    if not loaded:
                        st.warning("⚠️ Custom model weights not found at `model/image_model/image_model.pth`. Running with random weights — results will not be meaningful.")

                    tensor = IMAGE_TRANSFORM(image).unsqueeze(0)
                    with torch.no_grad():
                        logits = model(tensor)
                        probs = torch.softmax(logits, dim=1)[0]

                    # Class 0 = Real, Class 1 = AI
                    real_prob = probs[0].item()
                    ai_prob = probs[1].item()
                    ai_pct = ai_prob * 100
                    confidence = max(ai_pct, 100 - ai_pct)

                    label = "AI Generated" if ai_prob > 0.5 else "Human/Real"

                    render_result(label, confidence, ai_pct, {
                        "AI Score": f"{ai_pct:.2f}%",
                        "Confidence": f"{confidence:.2f}%",
                        "Real Prob": f"{real_prob*100:.2f}%",
                    })

                    st.progress(min(ai_pct / 100, 1.0))

                except Exception as e:
                    st.error(f"Image analysis error: {e}")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
AI Content Authenticity Detector| 2026 &nbsp;·&nbsp;
</div>
""", unsafe_allow_html=True)
