import streamlit as st
from PIL import Image
import sys
import os

# ── Path setup ─────────────────────────────────────────────
sys.path.append(os.path.dirname(__file__))
from Pipeline.image_pipeline import load_image_model, predict_image
from Pipeline.text_pipeline import load_text_model, ensemble_predict

# ── Page Config ────────────────────────────────────────────
st.set_page_config(page_title="AI Content Authenticity System", layout="wide")

# ── SESSION STATE ──────────────────────────────────────────
if "mode" not in st.session_state:
    st.session_state.mode = None

if "result" not in st.session_state:
    st.session_state.result = None

# ── UI CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
body {
    background-color: #F9FAFB;
}

.sidebar-icons {
    display: flex;
    flex-direction: column;
    gap: 20px;
    align-items: center;
}

.icon {
    font-size: 20px;
    padding: 10px;
    border-radius: 10px;
    cursor: pointer;
}

.input-box {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 20px;
    padding: 15px;
    width: 60%;
    margin: auto;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.05);
}

.result-box {
    width: 60%;
    margin: auto;
    margin-top: 20px;
    background: white;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #E5E7EB;
}

.center-title {
    text-align: center;
    font-size: 26px;
    font-weight: 600;
    margin-top: 40px;
}

.bottom-box {
    position: fixed;
    bottom: 20px;
    left: 35%;
    width: 30%;
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

# ── SIDEBAR (ICONS ONLY) ───────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-icons">', unsafe_allow_html=True)
    st.write("🏠")
    st.write("🕘")
    st.write("➕")
    st.write("⚙️")
    st.markdown('</div>', unsafe_allow_html=True)

# ── TITLE ──────────────────────────────────────────────────
st.markdown('<div class="center-title">AI Content Authenticity System</div>', unsafe_allow_html=True)

# ── INPUT FUNCTION ─────────────────────────────────────────
def input_ui(center=True):
    container_class = "input-box" if center else "bottom-box"

    st.markdown(f'<div class="{container_class}">', unsafe_allow_html=True)

    col1, col2 = st.columns([6,1])

    with col1:
        # + BUTTON OPTION
        option = st.selectbox(
            "",
            ["Select Input", "Text", "Image"],
            label_visibility="collapsed"
        )

        if option == "Text":
            user_text = st.text_area("", placeholder="Paste text...", height=100)
            st.session_state.mode = "text"
            return user_text, None

        elif option == "Image":
            uploaded_file = st.file_uploader("", type=["jpg","png","jpeg"])
            st.session_state.mode = "image"
            return None, uploaded_file

    with col2:
        analyze = st.button("→", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    return None, None if not analyze else None

# ── CENTER OR BOTTOM INPUT ─────────────────────────────────
if st.session_state.result is None:
    user_text, uploaded_file = input_ui(center=True)
else:
    user_text, uploaded_file = input_ui(center=False)

# ── PROCESS ────────────────────────────────────────────────
if st.button("Process", key="hidden_process", help="hidden trigger"):
    pass

if user_text or uploaded_file:
    if st.session_state.mode == "text" and user_text:
        with st.spinner("Processing..."):
            tokenizer, text_model = get_text_model()
            st.session_state.result = ensemble_predict(user_text, tokenizer, text_model)

    elif st.session_state.mode == "image" and uploaded_file:
        image = Image.open(uploaded_file)
        with st.spinner("Processing image..."):
            image_model = get_image_model()
            st.session_state.result = predict_image(image, image_model)

# ── RESULT CENTER ──────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    st.markdown('<div class="result-box">', unsafe_allow_html=True)

    st.subheader(r["label"])
    st.markdown(f"### {r['score']}%")
    st.write("AI Likelihood")

    col1, col2, col3 = st.columns(3)

    col1.metric("Confidence", f"{r['confidence']}%")

    if "roberta_ai" in r:
        col2.metric("RoBERTa", f"{r['roberta_ai']*100:.0f}%")
        col3.metric("Heuristic", f"{r['heuristic_ai']:.0f}%")
    else:
        col2.metric("AI Score", f"{r['score']}%")
        col3.metric("Real Probability", f"{round(r['real_probability']*100,2)}%")

    st.progress(r["ai_probability"])

    st.markdown('</div>', unsafe_allow_html=True)
