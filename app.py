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
    page_icon="🔍",
    layout="wide"
)

# ── PROFESSIONAL CLEAN UI ──────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Remove Streamlit default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    /* Header */
    .site-header {
        border-bottom: 1px solid #E5E7EB;
        padding-bottom: 1.2rem;
        margin-bottom: 1.5rem;
    }
    .site-title {
        font-size: 22px;
        font-weight: 700;
        color: #111827;
        letter-spacing: -0.3px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .site-subtitle {
        font-size: 13px;
        color: #9CA3AF;
        margin-top: 2px;
        font-weight: 400;
    }
    .badge {
        display: inline-block;
        font-size: 10px;
        font-weight: 600;
        background: #F3F4F6;
        color: #6B7280;
        padding: 2px 8px;
        border-radius: 99px;
        border: 1px solid #E5E7EB;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        vertical-align: middle;
        margin-left: 4px;
    }

    /* Cards */
    .card {
        background: #FFFFFF;
        padding: 22px 26px;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        margin-bottom: 16px;
    }
    .card-title {
        font-size: 13px;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 14px;
    }

    /* Result verdict boxes */
    .verdict-box {
        padding: 18px 22px;
        border-radius: 10px;
        margin-bottom: 16px;
    }
    .verdict-ai {
        background: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 4px solid #DC2626;
    }
    .verdict-human {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 4px solid #16A34A;
    }
    .verdict-uncertain {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 4px solid #F59E0B;
    }
    .verdict-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .verdict-label-ai { color: #DC2626; }
    .verdict-label-human { color: #16A34A; }
    .verdict-label-uncertain { color: #D97706; }
    .verdict-score {
        font-size: 38px;
        font-weight: 700;
        color: #111827;
        line-height: 1;
        font-family: 'DM Mono', monospace;
    }
    .verdict-desc {
        font-size: 12px;
        color: #6B7280;
        margin-top: 4px;
    }

    /* Metric pills */
    .metrics-row {
        display: flex;
        gap: 10px;
        margin-top: 14px;
        flex-wrap: wrap;
    }
    .metric-pill {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 10px 14px;
        flex: 1;
        min-width: 90px;
        text-align: center;
    }
    .metric-pill-label {
        font-size: 10px;
        color: #9CA3AF;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-pill-value {
        font-size: 18px;
        font-weight: 700;
        color: #111827;
        font-family: 'DM Mono', monospace;
        margin-top: 2px;
    }

    /* Progress bar section */
    .progress-section {
        margin-top: 14px;
    }
    .progress-header {
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        color: #6B7280;
        margin-bottom: 6px;
    }
    .progress-track {
        height: 6px;
        background: #F3F4F6;
        border-radius: 99px;
        overflow: hidden;
    }
    .progress-fill-ai {
        height: 100%;
        background: #DC2626;
        border-radius: 99px;
        transition: width 0.4s ease;
    }
    .progress-fill-human {
        height: 100%;
        background: #16A34A;
        border-radius: 99px;
    }
    .progress-fill-uncertain {
        height: 100%;
        background: #F59E0B;
        border-radius: 99px;
    }

    /* Explainability */
    .explain-section {
        margin-top: 14px;
        padding-top: 14px;
        border-top: 1px solid #F3F4F6;
    }
    .explain-title {
        font-size: 11px;
        font-weight: 600;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 10px;
    }
    .explain-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 8px;
    }
    .explain-key {
        font-size: 12px;
        color: #374151;
        width: 130px;
        flex-shrink: 0;
    }
    .explain-bar-track {
        flex: 1;
        height: 5px;
        background: #F3F4F6;
        border-radius: 99px;
        overflow: hidden;
    }
    .explain-bar-fill {
        height: 100%;
        border-radius: 99px;
    }
    .explain-val {
        font-size: 11px;
        font-family: 'DM Mono', monospace;
        color: #6B7280;
        width: 36px;
        text-align: right;
        flex-shrink: 0;
    }

    /* Image display — smaller, clean */
    .stImage img {
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        max-height: 260px;
        object-fit: cover;
        width: 100%;
    }

    /* Footer */
    .footer {
        margin-top: 2.5rem;
        padding-top: 1.2rem;
        border-top: 1px solid #F3F4F6;
        font-size: 12px;
        color: #9CA3AF;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Tab-like radio */
    div[data-testid="stHorizontalBlock"] { gap: 0; }
    div.stRadio > label { display: none; }
    div.stRadio > div {
        display: flex;
        gap: 0;
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 3px;
        width: fit-content;
    }
    div.stRadio > div > label {
        display: block !important;
        padding: 7px 20px;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 500;
        color: #6B7280;
        cursor: pointer;
        transition: 0.15s;
    }
    div.stRadio > div > label[data-selected="true"],
    div.stRadio > div > label:has(input:checked) {
        background: white;
        color: #111827;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }

    /* Streamlit button */
    .stButton > button {
        background: #111827;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 13px;
        font-weight: 600;
        font-family: 'DM Sans', sans-serif;
        width: 100%;
        cursor: pointer;
        transition: background 0.15s;
        margin-top: 8px;
    }
    .stButton > button:hover {
        background: #1F2937;
    }

    /* Textarea */
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        font-family: 'DM Sans', sans-serif;
        font-size: 13px;
        background: #FAFAFA;
    }
    .stTextArea textarea:focus {
        border-color: #9CA3AF;
        box-shadow: none;
    }

    /* File uploader */
    .stFileUploader {
        background: #F9FAFB;
        border: 1.5px dashed #D1D5DB;
        border-radius: 10px;
        padding: 10px;
    }

    /* Divider */
    hr { border-color: #F3F4F6; margin: 0.8rem 0; }

    /* Warning */
    .stAlert { border-radius: 8px; font-size: 13px; }
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
st.markdown("""
<div class="site-header">
    <div class="site-title">
        🔍 AI Content Authenticity Detector
        <span class="badge">Research v1.0</span>
    </div>
    <div class="site-subtitle">K.R. Mangalam University · MCA Capstone Project · 2025–26</div>
</div>
""", unsafe_allow_html=True)


# ── MODE SWITCH ────────────────────────────────────────────
mode = st.radio("", ["Text Analysis", "Image Analysis"], horizontal=True)
st.markdown("<br>", unsafe_allow_html=True)


# ── HELPER: determine verdict type ─────────────────────────
def get_verdict_type(confidence, label):
    if confidence < 60:
        return "uncertain"
    elif label == "AI Generated":
        return "ai"
    else:
        return "human"

def render_verdict(verdict_type, score, confidence, label):
    vclass = {
        "ai": "verdict-ai",
        "human": "verdict-human",
        "uncertain": "verdict-uncertain"
    }[verdict_type]

    vlabel_class = {
        "ai": "verdict-label-ai",
        "human": "verdict-label-human",
        "uncertain": "verdict-label-uncertain"
    }[verdict_type]

    display_label = {
        "ai": "AI Generated",
        "human": "Human Written",
        "uncertain": "Uncertain"
    }[verdict_type]

    desc = {
        "ai": "High likelihood this content is AI-generated.",
        "human": "Content appears to be human-authored.",
        "uncertain": f"Confidence ({confidence}%) is below threshold. Result inconclusive."
    }[verdict_type]

    progress_class = {
        "ai": "progress-fill-ai",
        "human": "progress-fill-human",
        "uncertain": "progress-fill-uncertain"
    }[verdict_type]

    display_score = score if verdict_type != "uncertain" else "–"

    st.markdown(f"""
    <div class="verdict-box {vclass}">
        <div class="verdict-label {vlabel_class}">{display_label}</div>
        <div class="verdict-score">{display_score}{'%' if verdict_type != 'uncertain' else ''}</div>
        <div class="verdict-desc">{desc}</div>
    </div>
    <div class="progress-section">
        <div class="progress-header">
            <span>AI Likelihood</span>
            <span>{score}%</span>
        </div>
        <div class="progress-track">
            <div class="{progress_class}" style="width:{score}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ───────────────────────── TEXT MODE ───────────────────────
if "Text" in mode:
    st.markdown('<div class="card"><div class="card-title">Text Detection</div>', unsafe_allow_html=True)

    user_text = st.text_area(
        "",
        height=160,
        placeholder="Paste your content here — minimum ~50 words recommended for accurate results."
    )

    analyze = st.button("Analyze Text")

    if analyze:
        if not user_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing..."):
                tokenizer, text_model = get_text_model()
                result = ensemble_predict(user_text, tokenizer, text_model)

            verdict = get_verdict_type(result["confidence"], result["label"])

            col_result, col_metrics = st.columns([1.2, 1])

            with col_result:
                render_verdict(verdict, result["score"], result["confidence"], result["label"])

            with col_metrics:
                st.markdown(f"""
                <div class="metrics-row">
                    <div class="metric-pill">
                        <div class="metric-pill-label">Confidence</div>
                        <div class="metric-pill-value">{result['confidence']}%</div>
                    </div>
                    <div class="metric-pill">
                        <div class="metric-pill-label">RoBERTa</div>
                        <div class="metric-pill-value">{result['roberta_ai']*100:.0f}%</div>
                    </div>
                    <div class="metric-pill">
                        <div class="metric-pill-label">Heuristic</div>
                        <div class="metric-pill-value">{result['heuristic_ai']:.0f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Explainability ──
                heuristic_ai = result.get("heuristic_ai", 50)
                roberta_ai   = result.get("roberta_ai", 0.5) * 100
                ensemble     = result["score"]

                # Compute visual proxy scores for each feature
                features = {
                    "RoBERTa model": roberta_ai,
                    "Heuristic analysis": heuristic_ai,
                    "Ensemble (final)": ensemble,
                }

                rows_html = ""
                for feat, val in features.items():
                    val = round(min(max(val, 0), 100))
                    color = "#DC2626" if val > 65 else "#F59E0B" if val > 40 else "#16A34A"
                    rows_html += f"""
                    <div class="explain-row">
                        <div class="explain-key">{feat}</div>
                        <div class="explain-bar-track">
                            <div class="explain-bar-fill" style="width:{val}%; background:{color}"></div>
                        </div>
                        <div class="explain-val">{val}%</div>
                    </div>"""

                st.markdown(f"""
                <div class="explain-section">
                    <div class="explain-title">Score Breakdown</div>
                    {rows_html}
                </div>
                """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ───────────────────────── IMAGE MODE ──────────────────────
if "Image" in mode:
    st.markdown('<div class="card"><div class="card-title">Image Detection</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)

        with st.spinner("Analyzing image..."):
            image_model = get_image_model()
            result = predict_image(image, image_model)

        verdict = get_verdict_type(result["confidence"], result["label"])

        # Side-by-side: image | analysis
        col_img, col_analysis = st.columns([1, 1.1], gap="large")

        with col_img:
            st.markdown('<div style="font-size:11px;color:#9CA3AF;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">Uploaded Image</div>', unsafe_allow_html=True)
            # Display image smaller using width param
            st.image(image, use_container_width=True)
            st.markdown(f'<div style="font-size:11px;color:#9CA3AF;margin-top:6px;">{uploaded_file.name} · {image.size[0]}×{image.size[1]}px</div>', unsafe_allow_html=True)

        with col_analysis:
            st.markdown('<div style="font-size:11px;color:#9CA3AF;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">Analysis Result</div>', unsafe_allow_html=True)

            render_verdict(verdict, result["score"], result["confidence"], result["label"])

            st.markdown(f"""
            <div class="metrics-row">
                <div class="metric-pill">
                    <div class="metric-pill-label">Confidence</div>
                    <div class="metric-pill-value">{result['confidence']}%</div>
                </div>
                <div class="metric-pill">
                    <div class="metric-pill-label">AI Score</div>
                    <div class="metric-pill-value">{result['score']}%</div>
                </div>
                <div class="metric-pill">
                    <div class="metric-pill-label">Real Prob.</div>
                    <div class="metric-pill-value">{round(result['real_probability']*100)}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <span>AI Content Authenticity Detection System</span>
    <span>For research & analysis purposes only</span>
</div>
""", unsafe_allow_html=True)
