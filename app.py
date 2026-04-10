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
    page_title="AI Content Authenticity System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── SESSION STATE ──────────────────────────────────────────
for key, default in {
    "result": None,
    "history": [],
    "input_mode": None,
    "has_content": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global reset ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #111111 !important;
    color: #E5E5E5 !important;
}

[data-testid="stSidebar"] {
    background-color: #1A1A1A !important;
    border-right: 0.5px solid #2A2A2A !important;
    min-width: 56px !important;
    max-width: 200px !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 12px 8px !important;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }

/* ── Topbar ── */
.topbar {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 20px;
    border-bottom: 0.5px solid #2A2A2A;
    background: #161616;
    margin-bottom: 0;
}
.topbar-title {
    font-size: 14px;
    font-weight: 500;
    color: #E5E5E5;
}
.topbar-sub {
    font-size: 12px;
    color: #666;
    margin-left: 4px;
}

/* ── Center hero ── */
.hero-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 55vh;
    gap: 10px;
    text-align: center;
}
.hero-icon {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    border: 1px solid #333;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 8px;
    font-size: 20px;
    background: #1E1E1E;
    color: #aaa;
}
.hero-title {
    font-size: 22px;
    font-weight: 500;
    color: #E5E5E5;
    margin: 0;
}
.hero-sub {
    font-size: 13px;
    color: #666;
    margin: 0;
}

/* ── Input box ── */
.input-wrap-outer {
    position: fixed;
    bottom: 20px;
    left: 50%;
    transform: translateX(-40%);
    width: 520px;
    background: #1E1E1E;
    border: 0.5px solid #333;
    border-radius: 14px;
    padding: 8px 12px;
    z-index: 999;
}

/* ── Result card ── */
.result-outer {
    max-width: 560px;
    margin: 20px auto;
    background: #1A1A1A;
    border: 0.5px solid #2A2A2A;
    border-radius: 14px;
    padding: 18px 20px;
}
.result-tag {
    font-size: 11px;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}
.result-verdict-ai {
    font-size: 20px;
    font-weight: 500;
    color: #E06B3C;
    margin-bottom: 10px;
}
.result-verdict-human {
    font-size: 20px;
    font-weight: 500;
    color: #3CB878;
    margin-bottom: 10px;
}
.conf-label {
    font-size: 12px;
    color: #666;
    margin-top: 4px;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: #E06B3C !important;
    border-radius: 3px !important;
}
.stProgress > div > div {
    background: #2A2A2A !important;
    border-radius: 3px !important;
    height: 6px !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #1E1E1E !important;
    border: 0.5px solid #2A2A2A !important;
    border-radius: 10px !important;
    padding: 12px !important;
}
[data-testid="stMetricLabel"] { color: #666 !important; font-size: 12px !important; }
[data-testid="stMetricValue"] { color: #E5E5E5 !important; font-size: 20px !important; }

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    border: 0.5px solid #333 !important;
    color: #E5E5E5 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    padding: 6px 14px !important;
}
.stButton > button:hover {
    background: #2A2A2A !important;
    border-color: #444 !important;
}

/* ── Primary action button ── */
.stButton > button[kind="primary"] {
    background: #533AB7 !important;
    border: none !important;
    color: white !important;
    border-radius: 8px !important;
}
.stButton > button[kind="primary"]:hover {
    background: #6245C8 !important;
}

/* ── Inputs / textarea ── */
.stTextArea textarea {
    background: #161616 !important;
    border: 0.5px solid #333 !important;
    border-radius: 8px !important;
    color: #E5E5E5 !important;
    font-size: 14px !important;
    resize: none !important;
}
.stTextArea textarea::placeholder { color: #555 !important; }
.stTextArea textarea:focus {
    border-color: #444 !important;
    box-shadow: none !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #161616 !important;
    border: 0.5px dashed #333 !important;
    border-radius: 10px !important;
    padding: 12px !important;
}
[data-testid="stFileUploader"] label { color: #666 !important; font-size: 13px !important; }

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #1E1E1E !important;
    border: 0.5px solid #333 !important;
    color: #E5E5E5 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
}

/* ── Sidebar history items ── */
.hist-item {
    padding: 7px 10px;
    border-radius: 7px;
    cursor: pointer;
    font-size: 12px;
    color: #888;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 2px;
}
.hist-item:hover { background: #222; color: #ccc; }

/* ── Sidebar icon buttons ── */
.sidebar-icon-btn {
    width: 36px;
    height: 36px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    color: #888;
    font-size: 16px;
    margin: 0 auto 4px;
    background: transparent;
    border: none;
}
.sidebar-icon-btn:hover { background: #222; color: #ccc; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #533AB7 !important; }

/* ── Divider ── */
hr { border-color: #2A2A2A !important; }

/* ── Image preview ── */
.img-preview {
    border-radius: 8px;
    border: 0.5px solid #2A2A2A;
    max-height: 140px;
    object-fit: cover;
}

/* ── Uploaded image caption chip ── */
.preview-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #222;
    border: 0.5px solid #333;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px;
    color: #888;
    margin-bottom: 6px;
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
    # New chat button
    if st.button("＋", help="New chat", use_container_width=False):
        st.session_state.result = None
        st.session_state.input_mode = None
        st.session_state.has_content = False
        st.rerun()

    st.markdown("<hr style='margin:8px 0'>", unsafe_allow_html=True)

    # History icon label
    st.markdown(
        "<div style='font-size:11px;color:#555;text-align:center;"
        "letter-spacing:0.5px;text-transform:uppercase;padding:4px 0'>🕘</div>",
        unsafe_allow_html=True
    )

    # History list
    if st.session_state.history:
        for item in st.session_state.history[-12:][::-1]:
            st.markdown(
                f"<div class='hist-item'>💬 {item}</div>",
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            "<div style='font-size:11px;color:#444;text-align:center;"
            "padding:8px 4px'>No history yet</div>",
            unsafe_allow_html=True
        )

# ── TOPBAR ─────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <span style="font-size:18px;color:#888">✦</span>
  <span class="topbar-title">AI Content Authenticity System</span>
  <span class="topbar-sub">— detect AI-generated text &amp; images</span>
</div>
""", unsafe_allow_html=True)

# ── HERO (shown only when no result yet) ───────────────────
if not st.session_state.has_content:
    st.markdown("""
    <div class="hero-wrap">
      <div class="hero-icon">✦</div>
      <p class="hero-title">AI Content Authenticity System</p>
      <p class="hero-sub">Upload an image or paste text to analyse</p>
    </div>
    """, unsafe_allow_html=True)

# ── RESULT DISPLAY ─────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result
    is_ai = "ai" in r.get("label", "").lower() or r.get("ai_probability", 0) > 0.5
    verdict_class = "result-verdict-ai" if is_ai else "result-verdict-human"
    label_display = r.get("label", "Unknown")
    score = r.get("score", 0)
    confidence = r.get("confidence", score)
    ai_prob = r.get("ai_probability", score / 100)

    st.markdown(f"""
    <div class="result-outer">
      <div class="result-tag">Analysis result</div>
      <div class="{verdict_class}">{label_display}</div>
    </div>
    """, unsafe_allow_html=True)

    result_col1, result_col2, result_col3 = st.columns(3)
    result_col1.metric("Confidence", f"{confidence}%")

    if "roberta_ai" in r:
        result_col2.metric("RoBERTa", f"{r['roberta_ai'] * 100:.0f}%")
        result_col3.metric("Heuristic", f"{r['heuristic_ai']:.0f}%")
    else:
        result_col2.metric("AI score", f"{score}%")
        result_col3.metric("Real probability", f"{round(r.get('real_probability', 0) * 100, 2)}%")

    st.markdown(
        f"<div style='font-size:12px;color:#666;margin:6px 0 4px'>AI likelihood</div>",
        unsafe_allow_html=True
    )
    st.progress(float(ai_prob))

    st.markdown("<div style='height:80px'></div>", unsafe_allow_html=True)

# ── INPUT PANEL (fixed bottom) ─────────────────────────────
st.markdown("<div style='height:100px'></div>", unsafe_allow_html=True)

st.markdown("""
<div style='position:fixed;bottom:0;left:0;right:0;height:1px;
background:linear-gradient(90deg,transparent,#2A2A2A,transparent)'></div>
""", unsafe_allow_html=True)

input_container = st.container()
with input_container:
    st.markdown(
        "<div style='position:fixed;bottom:0;left:56px;right:0;"
        "background:#111;padding:12px 20px 18px;border-top:0.5px solid #2A2A2A;z-index:998'>",
        unsafe_allow_html=True
    )

    col_plus, col_mode, col_main, col_send = st.columns([0.5, 1.5, 6, 0.8])

    with col_plus:
        show_menu = st.button("＋", key="plus_btn", help="Choose input type")

    with col_mode:
        input_type = st.selectbox(
            "",
            ["Text", "Image"],
            key="input_type_select",
            label_visibility="collapsed"
        )

    user_text = None
    uploaded_file = None

    with col_main:
        if input_type == "Text":
            user_text = st.text_area(
                "",
                placeholder="Paste text to analyse...",
                height=60,
                key="text_input",
                label_visibility="collapsed"
            )
            st.session_state.input_mode = "text"
        else:
            uploaded_file = st.file_uploader(
                "",
                type=["jpg", "png", "jpeg"],
                key="image_input",
                label_visibility="collapsed"
            )
            st.session_state.input_mode = "image"

    with col_send:
        analyse_clicked = st.button("→", key="analyse_btn", type="primary")

    st.markdown("</div>", unsafe_allow_html=True)

# ── PROCESS ────────────────────────────────────────────────
if analyse_clicked:
    if st.session_state.input_mode == "text" and user_text and user_text.strip():
        st.session_state.has_content = True
        snippet = user_text.strip()[:35] + ("…" if len(user_text.strip()) > 35 else "")
        st.session_state.history.append(snippet)

        with st.spinner("Analysing text…"):
            tokenizer, text_model = get_text_model()
            st.session_state.result = ensemble_predict(
                user_text.strip(), tokenizer, text_model
            )
        st.rerun()

    elif st.session_state.input_mode == "image" and uploaded_file is not None:
        st.session_state.has_content = True
        st.session_state.history.append(f"Image: {uploaded_file.name}")

        image = Image.open(uploaded_file)
        with st.spinner("Analysing image…"):
            image_model = get_image_model()
            st.session_state.result = predict_image(image, image_model)
        st.rerun()

    else:
        st.warning("Please provide text or upload an image before analysing.")
