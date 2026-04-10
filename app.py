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
    initial_sidebar_state="expanded"
)

# ── STATE ─────────────────────────────────
for k, v in {
    "result": None,
    "history": [],
    "input_mode": "text",
    "has_content": False,
    "show_menu": False
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── CLEAN RESPONSIVE CSS ───────────────────
st.markdown("""
<style>
html, body, .stApp {
    background: #0F0F0F;
    color: #D4D4D4;
}

/* FIX HEIGHT ISSUE */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* TOP BAR */
.topbar {
    height: 48px;
    display: flex;
    align-items: center;
    padding: 0 16px;
    border-bottom: 1px solid #1E1E1E;
    background: #111;
    font-size: 14px;
}

/* HERO CENTER */
.hero {
    height: calc(100vh - 120px);
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
}

/* RESULT */
.result {
    max-width: 600px;
    margin: 20px auto;
}

/* INPUT BAR */
.input-bar {
    position: fixed;
    bottom: 0;
    width: 100%;
    padding: 10px 16px;
    background: #0F0F0F;
    border-top: 1px solid #1E1E1E;
}

/* TEXTAREA FIX */
textarea {
    font-size: 14px !important;
    line-height: 1.5 !important;
}

/* BUTTONS */
.stButton > button {
    border-radius: 8px;
}

/* ROUND BUTTONS */
div[data-testid="column"]:nth-child(1) button,
div[data-testid="column"]:nth-child(3) button {
    width: 40px;
    height: 40px;
    border-radius: 50%;
}

/* SEND BUTTON */
div[data-testid="column"]:nth-child(3) button {
    background: #533AB7 !important;
    color: white !important;
}

/* POPUP */
.menu {
    position: fixed;
    bottom: 70px;
    left: 20px;
    background: #1A1A1A;
    padding: 10px;
    border-radius: 10px;
    border: 1px solid #2A2A2A;
}

/* SIDEBAR FIX */
section[data-testid="stSidebarCollapsedControl"] {
    top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ── MODELS ─────────────────────────────────
@st.cache_resource
def get_image_model():
    return load_image_model("model/image_model/image_model.pth")

@st.cache_resource
def get_text_model():
    return load_text_model("model/text_model")

# ── SIDEBAR ────────────────────────────────
with st.sidebar:
    if st.button("New Chat"):
        st.session_state.result = None
        st.session_state.has_content = False
        st.rerun()

    st.write("History")
    for h in st.session_state.history[-10:]:
        st.write(h)

# ── TOP BAR ────────────────────────────────
st.markdown('<div class="topbar">AI Content Authenticity System</div>', unsafe_allow_html=True)

# ── MAIN ───────────────────────────────────
if not st.session_state.has_content:
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.subheader("AI Content Authenticity System")
    st.caption("Upload image or paste text")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    if st.session_state.result:
        r = st.session_state.result
        st.markdown('<div class="result">', unsafe_allow_html=True)
        st.subheader(r["label"])
        st.write(f"Score: {r['score']}%")
        st.progress(r["ai_probability"])
        st.markdown('</div>', unsafe_allow_html=True)

# ── POPUP MENU ─────────────────────────────
if st.session_state.show_menu:
    st.markdown('<div class="menu">', unsafe_allow_html=True)
    if st.button("Text"):
        st.session_state.input_mode = "text"
        st.session_state.show_menu = False
        st.rerun()
    if st.button("Image"):
        st.session_state.input_mode = "image"
        st.session_state.show_menu = False
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ── INPUT BAR ──────────────────────────────
st.markdown('<div class="input-bar">', unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 8, 1])

with c1:
    if st.button("+"):
        st.session_state.show_menu = not st.session_state.show_menu
        st.rerun()

with c2:
    if st.session_state.input_mode == "text":
        user_text = st.text_area("", placeholder="Paste text...", height=40)
        uploaded = None
    else:
        uploaded = st.file_uploader("", type=["jpg","png","jpeg"])
        user_text = None

with c3:
    send = st.button("→")

st.markdown('</div>', unsafe_allow_html=True)

# ── PROCESS ────────────────────────────────
if send:
    if st.session_state.input_mode == "text" and user_text:
        st.session_state.history.append(user_text[:30])
        st.session_state.has_content = True
        tokenizer, model = get_text_model()
        st.session_state.result = ensemble_predict(user_text, tokenizer, model)
        st.rerun()

    elif st.session_state.input_mode == "image" and uploaded:
        st.session_state.history.append(uploaded.name)
        st.session_state.has_content = True
        model = get_image_model()
        st.session_state.result = predict_image(Image.open(uploaded), model)
        st.rerun()

    else:
        st.warning("Enter text or upload image")
