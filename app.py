import streamlit as st
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(__file__))
from Pipeline.image_pipeline import load_image_model, predict_image
from Pipeline.text_pipeline import load_text_model, ensemble_predict

st.set_page_config(
    page_title="AI Content Authenticity System",
    layout="wide",
    initial_sidebar_state="collapsed",
)

for key, default in {
    "result": None,
    "history": [],
    "input_mode": "text",
    "has_content": False,
    "show_input_menu": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.markdown("""
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="block-container"] {
    background: #0F0F0F !important;
    color: #D4D4D4 !important;
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
    padding: 0 !important;
    max-width: 100% !important;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #252525; border-radius: 2px; }

*, div, span, p, label, button {
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif !important;
    -webkit-font-smoothing: antialiased !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #141414 !important;
    border-right: 0.5px solid #1E1E1E !important;
    min-width: 190px !important;
    max-width: 190px !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 10px 8px !important; }

section[data-testid="stSidebarCollapsedControl"] {
    background: transparent !important;
    top: 10px !important;
}
section[data-testid="stSidebarCollapsedControl"] button {
    background: #1A1A1A !important;
    border: 0.5px solid #252525 !important;
    border-radius: 7px !important;
    color: #666 !important;
}
section[data-testid="stSidebarCollapsedControl"] button:hover {
    background: #222 !important;
    color: #aaa !important;
}

/* Topbar */
.topbar {
    display: flex;
    align-items: center;
    gap: 8px;
    height: 44px;
    padding: 0 20px;
    border-bottom: 0.5px solid #1E1E1E;
    background: #111;
}
.topbar-title {
    font-size: 13px;
    font-weight: 500;
    color: #C8C8C8;
    letter-spacing: -0.01em;
}
.topbar-sub { font-size: 12px; color: #3A3A3A; }

/* Hero */
.hero-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: calc(100vh - 44px - 76px);
    gap: 10px;
    text-align: center;
    padding: 16px;
}
.hero-icon {
    width: 44px; height: 44px;
    border-radius: 50%;
    border: 0.5px solid #252525;
    background: #181818;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 6px;
    font-size: 18px; color: #444;
}
.hero-title {
    font-size: 20px; font-weight: 500;
    color: #C8C8C8; letter-spacing: -0.02em; line-height: 1.3;
}
.hero-sub { font-size: 13px; color: #3A3A3A; line-height: 1.5; }

/* Result scroll area */
.result-scroll-area {
    height: calc(100vh - 44px - 76px);
    overflow-y: auto;
    padding: 18px 20px 10px;
}
.result-card {
    max-width: 500px;
    margin: 0 auto 14px;
    background: #161616;
    border: 0.5px solid #222;
    border-radius: 12px;
    padding: 14px 16px;
}
.result-tag {
    font-size: 10px; color: #3A3A3A;
    text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 6px;
}
.verdict-ai { font-size: 17px; font-weight: 500; color: #C86040; margin-bottom: 6px; }
.verdict-human { font-size: 17px; font-weight: 500; color: #3A9A68; margin-bottom: 6px; }
.conf-label { font-size: 11px; color: #3A3A3A; }

/* Progress */
.stProgress > div > div {
    background: #1E1E1E !important; border-radius: 3px !important; height: 4px !important;
}
.stProgress > div > div > div {
    background: #533AB7 !important; border-radius: 3px !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #161616 !important;
    border: 0.5px solid #222 !important;
    border-radius: 9px !important;
    padding: 10px 12px !important;
}
[data-testid="stMetricLabel"] p { color: #444 !important; font-size: 11px !important; }
[data-testid="stMetricValue"] {
    color: #C8C8C8 !important; font-size: 17px !important; font-weight: 500 !important;
}

/* Bottom bar */
.btm-bar {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    background: #0F0F0F;
    border-top: 0.5px solid #1A1A1A;
    padding: 9px 16px 13px;
    z-index: 999;
}

/* Input popup */
.input-popup {
    position: fixed;
    bottom: 72px; left: 16px;
    background: #191919;
    border: 0.5px solid #252525;
    border-radius: 10px;
    padding: 5px;
    min-width: 145px;
    z-index: 1001;
    box-shadow: 0 8px 28px rgba(0,0,0,0.5);
}
.popup-item {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 9px; border-radius: 7px;
    font-size: 12px; color: #B0B0B0; cursor: pointer;
}
.popup-item:hover { background: #222; }
.pop-icon {
    width: 22px; height: 22px;
    border-radius: 5px;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px;
}

/* All buttons base */
.stButton > button {
    background: #191919 !important;
    border: 0.5px solid #252525 !important;
    color: #B0B0B0 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    height: 40px !important;
    transition: all 0.12s !important;
    letter-spacing: 0 !important;
}
.stButton > button:hover {
    background: #222 !important;
    border-color: #303030 !important;
    color: #D4D4D4 !important;
}

/* Plus button — first col */
div[data-testid="column"]:nth-child(1) .stButton > button {
    width: 40px !important; height: 40px !important;
    padding: 0 !important; border-radius: 50% !important;
    font-size: 18px !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
}

/* Send button — last col */
div[data-testid="column"]:nth-child(3) .stButton > button {
    background: #533AB7 !important;
    border: none !important;
    color: #fff !important;
    border-radius: 50% !important;
    width: 40px !important; height: 40px !important;
    padding: 0 !important; font-size: 15px !important;
}
div[data-testid="column"]:nth-child(3) .stButton > button:hover {
    background: #6245C8 !important;
}

/* Textarea */
.stTextArea { margin: 0 !important; }
.stTextArea > div { margin: 0 !important; }
.stTextArea textarea {
    background: #141414 !important;
    border: 0.5px solid #222 !important;
    border-radius: 8px !important;
    color: #C8C8C8 !important;
    font-size: 13px !important;
    line-height: 1.55 !important;
    resize: none !important;
    padding: 9px 11px !important;
    min-height: 40px !important;
    max-height: 120px !important;
}
.stTextArea textarea::placeholder { color: #303030 !important; }
.stTextArea textarea:focus {
    border-color: #2E2E2E !important;
    box-shadow: none !important; outline: none !important;
}
.stTextArea label { display: none !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #141414 !important;
    border: 0.5px dashed #222 !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: #141414 !important;
    padding: 6px !important;
    min-height: 40px !important;
}
[data-testid="stFileUploaderDropzone"] span { color: #303030 !important; font-size: 12px !important; }
[data-testid="stFileUploader"] label { display: none !important; }

/* Sidebar history */
.hist-hdr {
    font-size: 10px; color: #282828;
    text-transform: uppercase; letter-spacing: 0.6px;
    padding: 0 6px 5px;
}
.hist-row {
    padding: 6px 8px; border-radius: 6px;
    font-size: 12px; color: #555; cursor: pointer;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.hist-row:hover { background: #1A1A1A; color: #999; }

/* Warning */
[data-testid="stAlert"] {
    background: #161608 !important;
    border-color: #2A2A10 !important;
    color: #666 !important;
    font-size: 12px !important;
    border-radius: 8px !important;
}

/* Spinner */
.stSpinner > div { border-top-color: #533AB7 !important; }

/* Column alignment */
[data-testid="stHorizontalBlock"] {
    gap: 8px !important;
    align-items: center !important;
}
[data-testid="stVerticalBlock"] { gap: 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── MODELS ─────────────────────────────────────────────────
@st.cache_resource
def get_image_model():
    return load_image_model("model/image_model/image_model.pth")

@st.cache_resource
def get_text_model():
    return load_text_model("model/text_model")


# ── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    if st.button("＋  New chat", use_container_width=True, key="new_chat"):
        st.session_state.result = None
        st.session_state.has_content = False
        st.session_state.show_input_menu = False
        st.rerun()

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='hist-hdr'>History</div>", unsafe_allow_html=True)

    if st.session_state.history:
        for item in reversed(st.session_state.history[-15:]):
            st.markdown(f"<div class='hist-row'>💬 {item}</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='font-size:11px;color:#252525;padding:6px 8px'>No sessions yet</div>",
            unsafe_allow_html=True
        )


# ── TOPBAR ─────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <span style="font-size:15px;color:#3A3A3A">✦</span>
  <span class="topbar-title">AI Content Authenticity System</span>
  <span class="topbar-sub">&nbsp;— detect AI-generated text &amp; images</span>
</div>
""", unsafe_allow_html=True)


# ── MAIN AREA ──────────────────────────────────────────────
if not st.session_state.has_content:
    st.markdown("""
    <div class="hero-wrap">
      <div class="hero-icon">✦</div>
      <div class="hero-title">AI Content Authenticity System</div>
      <div class="hero-sub">Upload an image or paste text to analyse</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("<div class='result-scroll-area'>", unsafe_allow_html=True)

    if st.session_state.result:
        r = st.session_state.result
        is_ai = "ai" in r.get("label", "").lower() or r.get("ai_probability", 0) > 0.5
        vc = "verdict-ai" if is_ai else "verdict-human"
        score = r.get("score", 0)
        confidence = r.get("confidence", score)
        ai_prob = float(r.get("ai_probability", score / 100))

        st.markdown(f"""
        <div class="result-card">
          <div class="result-tag">Analysis result</div>
          <div class="{vc}">{r.get("label","Unknown")}</div>
          <div class="conf-label">AI likelihood &nbsp;·&nbsp; {score}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(ai_prob)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence", f"{confidence}%")
        if "roberta_ai" in r:
            c2.metric("RoBERTa", f"{r['roberta_ai']*100:.0f}%")
            c3.metric("Heuristic", f"{r['heuristic_ai']:.0f}%")
        else:
            c2.metric("AI score", f"{score}%")
            c3.metric("Real prob.", f"{round(r.get('real_probability',0)*100,1)}%")

    st.markdown("</div>", unsafe_allow_html=True)


# ── INPUT POPUP (above bar) ────────────────────────────────
if st.session_state.show_input_menu:
    st.markdown("""
    <div class="input-popup">
      <div class="popup-item">
        <div class="pop-icon" style="background:#0E2235">🖼</div> Upload image
      </div>
      <div class="popup-item">
        <div class="pop-icon" style="background:#0E2520">📝</div> Paste text
      </div>
    </div>
    """, unsafe_allow_html=True)

    pc1, pc2, _ = st.columns([1.4, 1.4, 5])
    with pc1:
        if st.button("🖼 Image", key="pick_image"):
            st.session_state.input_mode = "image"
            st.session_state.show_input_menu = False
            st.rerun()
    with pc2:
        if st.button("📝 Text", key="pick_text"):
            st.session_state.input_mode = "text"
            st.session_state.show_input_menu = False
            st.rerun()


# ── BOTTOM INPUT BAR ───────────────────────────────────────
st.markdown("<div class='btm-bar'>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([0.55, 9, 0.55])

with col1:
    if st.button("＋", key="plus_btn"):
        st.session_state.show_input_menu = not st.session_state.show_input_menu
        st.rerun()

with col2:
    if st.session_state.input_mode == "text":
        user_text = st.text_area(
            "txt",
            placeholder="Paste text to analyse…",
            height=42,
            key="txt_in",
            label_visibility="collapsed",
        )
        uploaded_file = None
    else:
        uploaded_file = st.file_uploader(
            "img",
            type=["jpg", "png", "jpeg"],
            key="img_in",
            label_visibility="collapsed",
        )
        user_text = None

with col3:
    send = st.button("→", key="send_btn")

st.markdown("</div>", unsafe_allow_html=True)


# ── PROCESS ────────────────────────────────────────────────
if send:
    if st.session_state.input_mode == "text" and user_text and user_text.strip():
        snippet = user_text.strip()[:38] + ("…" if len(user_text.strip()) > 38 else "")
        st.session_state.history.append(snippet)
        st.session_state.has_content = True
        st.session_state.show_input_menu = False
        with st.spinner("Analysing text…"):
            tokenizer, text_model = get_text_model()
            st.session_state.result = ensemble_predict(user_text.strip(), tokenizer, text_model)
        st.rerun()

    elif st.session_state.input_mode == "image" and uploaded_file is not None:
        st.session_state.history.append(f"Image: {uploaded_file.name}")
        st.session_state.has_content = True
        st.session_state.show_input_menu = False
        with st.spinner("Analysing image…"):
            image_model = get_image_model()
            st.session_state.result = predict_image(Image.open(uploaded_file), image_model)
        st.rerun()

    else:
        st.warning("Please add text or upload an image first.")
