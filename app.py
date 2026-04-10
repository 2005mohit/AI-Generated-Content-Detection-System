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
    page_title="AI Content Authenticity Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── PROFESSIONAL UI STYLING ────────────────────────────────
st.markdown("""
<style>
    /* Global Styles */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stHeader"] {
        background: transparent;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Header */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        color: white;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        opacity: 0.95;
        font-weight: 300;
    }
    
    /* Main Container */
    .main-container {
        background: white;
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem auto;
        max-width: 1200px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
    }
    
    /* Mode Selector */
    .stRadio > div {
        background: #F8F9FA;
        padding: 0.5rem;
        border-radius: 12px;
        justify-content: center;
    }
    
    /* Text Area */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #E0E0E0;
        padding: 1rem;
        font-size: 1rem;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Result Cards */
    .result-card {
        padding: 2rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    .result-ai {
        background: linear-gradient(135deg, #FEE2E2 0%, #FCA5A5 100%);
        border: 3px solid #EF4444;
    }
    
    .result-human {
        background: linear-gradient(135deg, #D1FAE5 0%, #6EE7B7 100%);
        border: 3px solid #10B981;
    }
    
    .result-uncertain {
        background: linear-gradient(135deg, #FEF3C7 0%, #FCD34D 100%);
        border: 3px solid #F59E0B;
    }
    
    .result-label {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .result-score {
        font-size: 4rem;
        font-weight: 800;
        margin: 0.5rem 0;
        line-height: 1;
    }
    
    .result-description {
        font-size: 1rem;
        opacity: 0.8;
        margin-top: 0.5rem;
    }
    
    /* Metrics */
    .metric-box {
        background: #F8F9FA;
        padding: 1.2rem;
        border-radius: 12px;
        border: 2px solid #E9ECEF;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6C757D;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #212529;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #10B981 0%, #F59E0B 50%, #EF4444 100%);
        border-radius: 10px;
    }
    
    /* File Uploader */
    [data-testid="stFileUploadDropzone"] {
        border: 3px dashed #667eea;
        border-radius: 16px;
        padding: 2rem;
        background: #F8F9FA;
    }
    
    /* Image Display */
    [data-testid="stImage"] {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #E0E0E0, transparent);
    }
    
    /* Footer */
    .custom-footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: white;
        font-size: 0.9rem;
        opacity: 0.8;
    }
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
<div class="main-header">
    <div class="main-title">🔍 AI Content Authenticity Detector</div>
    <div class="main-subtitle">Detect AI-generated text and images with advanced machine learning models</div>
</div>
""", unsafe_allow_html=True)

# ── MAIN CONTAINER ─────────────────────────────────────────
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# ── MODE SELECTOR ──────────────────────────────────────────
mode = st.radio(
    "Select Analysis Mode:",
    ["📝 Text Detection", "🖼️ Image Detection"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# TEXT DETECTION MODE
# ═══════════════════════════════════════════════════════════
if "Text" in mode:
    
    st.markdown("### 📝 Text Analysis")
    st.markdown("Paste your text below to check if it's AI-generated or human-written.")
    
    user_text = st.text_area(
        "",
        height=200,
        placeholder="Enter your text here...\n\nExample: Articles, essays, emails, social media posts, or any written content.",
        label_visibility="collapsed"
    )
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyze = st.button(" Analyze Text", use_container_width=True)
    
    if analyze:
        if not user_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner(" Analyzing your text..."):
                tokenizer, text_model = get_text_model()
                result = ensemble_predict(user_text, tokenizer, text_model)
            
            # Determine result class based on score
            score = result['score']
            
            if score < 40:
                box_class = "result-human"
                label = " Human Written"
                
            elif score > 60:
                box_class = "result-ai"
                label = " AI Generated"
               
            else:
                box_class = "result-uncertain"
                label = " Uncertain"
            
            # Display result
            st.markdown(f"""
            <div class="result-card {box_class}">
                <div class="result-label">{label}</div>
                <div class="result-score">{score}%</div>
                <div class="result-description">AI Likelihood Score</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            st.markdown("#### Detailed Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{result['confidence']}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">RoBERTa Model</div>
                    <div class="metric-value">{result['roberta_ai']*100:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Heuristic Analysis</div>
                    <div class="metric-value">{result['heuristic_ai']:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Final Verdict</div>
                    <div class="metric-value" style="font-size: 1.3rem;">{emoji}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Progress bar
            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(result['ai_probability'], text=f"AI Probability: {score}%")
            
            # Interpretation
            st.markdown("---")
            st.markdown("#### Interpretation")
            if score < 40:
                st.success("This text appears to be **human-written**. The linguistic patterns, style variations, and structure suggest authentic human authorship.")
            elif score > 60:
                st.error("This text appears to be **AI-generated**. The text shows characteristics commonly found in machine-generated content such as consistent tone and predictable patterns.")
            else:
                st.warning("The result is **uncertain**. The text contains mixed signals that make it difficult to classify definitively. It could be human-written, AI-generated, or a combination of both.")


# ═══════════════════════════════════════════════════════════
# IMAGE DETECTION MODE
# ═══════════════════════════════════════════════════════════
if "Image" in mode:
    
    st.markdown("### 🖼️ Image Analysis")
    st.markdown("Upload an image to check if it's AI-generated or real.")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        # Analyze automatically
        with st.spinner(" Analyzing your image..."):
            image_model = get_image_model()
            result = predict_image(image, image_model)
        
        # Layout: Image on left, Results on right
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("#### Uploaded Image")
            # Display with fixed width for zoom-out effect
            st.image(image, use_column_width=True)
        
        with col2:
            # Determine result class
            score = result['score']
            
            if score < 40:
                box_class = "result-human"
                label = " Real Image"
            elif score > 60:
                box_class = "result-ai"
                label = " AI Generated"
            else:
                box_class = "result-uncertain"
                label = " Uncertain"
            
            # Display result
            st.markdown(f"""
            <div class="result-card {box_class}">
                <div class="result-label">{label}</div>
                <div class="result-score">{score}%</div>
                <div class="result-description">AI Likelihood Score</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            st.markdown("#### Detailed Metrics")
            colA, colB, colC = st.columns(3)
            
            with colA:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{result['confidence']}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with colB:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">AI Score</div>
                    <div class="metric-value">{score}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with colC:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Real Probability</div>
                    <div class="metric-value">{round(result['real_probability']*100,1)}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Progress bar
            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(result['ai_probability'], text=f"AI Probability: {score}%")
        
        # Interpretation (full width below)
        st.markdown("---")
        st.markdown("#### Interpretation")
        if score < 40:
            st.success("This image appears to be **real/authentic**. The visual patterns, textures, and structural elements suggest it was captured by a camera or created by a human artist.")
        elif score > 60:
            st.error("This image appears to be **AI-generated**. The image shows characteristics commonly found in synthetic content such as unusual textures, lighting inconsistencies, or anatomical artifacts.")
        else:
            st.warning("The result is **uncertain**. The image contains mixed signals that make classification difficult. It could be a real image, AI-generated, or heavily edited.")

st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────────
st.markdown("""
<div class="custom-footer">
    AI Content Detection System © 2025
    <br>
    <small>For research and educational purposes</small>
</div>
""", unsafe_allow_html=True)
