import streamlit as st
from PIL import Image
import sys
import os

# ── Path setup ───────────────────────────────────────────────────────────────
sys.path.append(os.path.dirname(__file__))
from pipeline.image_pipeline import load_image_model, predict_image
from pipeline.text_pipeline import load_text_model, predict_text

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Content Authenticity app",
    page_icon="",
    layout="centered"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .result-box {
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        text-align: center;
    }
    .ai-box    { background: #ffe5e5; border: 2px solid #ff4444; }
    .human-box { background: #e5ffe5; border: 2px solid #44bb44; }
    .metric-label { font-size: 14px; color: #666; margin-bottom: 4px; }
    .metric-value { font-size: 28px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Load Models (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def get_image_model():
    return load_image_model("model/image_model/image_model.pth")

@st.cache_resource
def get_text_model():
    return load_text_model("model/text_model")


# ── UI ────────────────────────────────────────────────────────────────────────
st.title(" AI Content Authenticity Detector")
st.markdown("Detect whether **text** or **image** is **AI-Generated** or **Human/Real**")
st.divider()

tab1, tab2 = st.tabs([" Text Detection", " Image Detection"])

# ─────────────────────────────────────────────────────
# TAB 1 — TEXT
# ─────────────────────────────────────────────────────
with tab1:
    st.subheader("Text AI Detection")
    st.caption("Uses RoBERTa (chatgpt-detector-roberta) model")

    user_text = st.text_area(
        "Enter text to analyze:",
        height=200,
        placeholder="Paste any article, essay, email, or content here..."
    )

    if st.button("🔍 Analyze Text", use_container_width=True, key="text_btn"):
        if not user_text.strip():
            st.warning("Please enter some text first!")
        else:
            with st.spinner("Analyzing text..."):
                tokenizer, text_model = get_text_model()
                result = predict_text(user_text, tokenizer, text_model)

            # Result display
            box_class = "ai-box" if result["label"] == "AI Generated" else "human-box"
            icon = "🤖" if result["label"] == "AI Generated" else ""

            st.markdown(f"""
            <div class="result-box {box_class}">
                <h2>{icon} {result['label']}</h2>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(" AI Score", f"{result['score']}%")
            with col2:
                st.metric(" Confidence", f"{result['confidence']}%")
            with col3:
                st.metric(" Human Prob", f"{round(result['human_probability']*100, 2)}%")

            st.divider()
            st.progress(result['ai_probability'], text=f"AI Likelihood: {result['score']}%")


# ─────────────────────────────────────────────────────
# TAB 2 — IMAGE
# ─────────────────────────────────────────────────────
with tab2:
    st.subheader("Image AI Detection")
    st.caption("Uses EfficientNet model")

    uploaded_file = st.file_uploader(
        "Upload an image:",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button(" Analyze Image", use_container_width=True, key="img_btn"):
            with st.spinner("Analyzing image..."):
                image_model = get_image_model()
                result = predict_image(image, image_model)

            # Result display
            box_class = "ai-box" if result["label"] == "AI Generated" else "human-box"
            icon = "🤖" if result["label"] == "AI Generated" else "📷"

            st.markdown(f"""
            <div class="result-box {box_class}">
                <h2>{icon} {result['label']}</h2>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(" AI Score", f"{result['score']}%")
            with col2:
                st.metric(" Confidence", f"{result['confidence']}%")
            with col3:
                st.metric("📷 Real Prob", f"{round(result['real_probability']*100, 2)}%")

            st.divider()
            st.progress(result['ai_probability'], text=f"AI Likelihood: {result['score']}%")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with  by 2005mohit | AI Content Authenticity System")
