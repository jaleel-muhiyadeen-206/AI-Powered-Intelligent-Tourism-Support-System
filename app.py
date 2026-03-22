# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
from PIL import Image
from classifier import get_prediction

# Page config
st.set_page_config(
    page_title="Sri Lanka Landmark Recognition",
    page_icon="🏛️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #020808 0%, #0a1f1f 50%, #031212 100%);
        min-height: 100vh;
    }

    .main-title {
        font-size: 1.5rem;  
        font-weight: bold;
        text-align: center;
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        margin: 5rem 0 0.6rem 0 !important;
        padding: 0 !important;
    }

    .subtitle {
        text-align: center;
        color: #ede4e4;
        font-size: 0rem;  
        margin: 0rem 0 2.5rem 0 !important;
        padding: 0 !important;
    }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 13px;
        padding: 1.5rem;
        color: white;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }

    .upload-section {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 0rem !important;  
        backdrop-filter: blur(10px);
        margin: 1rem 0 !important; 
    }
    
    /* Remove ALL padding/margin from uploader */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem 1rem !important;  
        margin: 0.5rem 0 !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease-in-out;
    }

    .stFileUploader:hover {
        background: rgba(255, 255, 255, 0.12);  
        border: 1px solid rgba(255, 255, 255, 0.25);
        box-shadow: 0 4px 20px rgba(255, 255, 255, 0.1);
    }

    .stFileUploader label {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
        padding-top: 0.5rem important;
        padding-bottom: 0.5rem important;
    }

    [data-testid="stFileUploader"] {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
        padding-top: 0.5rem important;
        padding-bottom: 0.5rem important;
    }

    [data-testid="stFileUploader"] label {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
        padding-top: 0.5rem important;
        padding-bottom: 0.5rem important;
    }

    /* Footer - full width */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100vw;
        text-align: center;
        color: #666666;
        font-size: 1rem;
        padding: 1rem;
        border-top: 1px solid rgba(255,255,255,0.08);
        background: rgba(10, 10, 15, 0.95);
        margin-left: calc(-50vw + 50%);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {visibility: hidden;}

    /* Minimize block container padding */
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 4rem !important;
    }

    /* Remove all element margins */
    .element-container {
        margin-top: 0rem !important;
        padding-top: 0rem !important;
    }

    /* Remove section margins */
    section {
        margin-top: 0rem !important;
        padding-top: 0rem !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TITLE SECTION - NO EXTRA SPACING
# -----------------------------------------------------------------------------
st.markdown('<h2 class="main-title">🏛️ Sri Lanka Landmark Recognition</h2>', unsafe_allow_html=True)
st.markdown('<h5 class="subtitle">Discover the beauty of Sri Lanka with AI-powered image recognition</h5>',
            unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------------------------
try:
    from classifier import init_classifier
    init_classifier()
    st.sidebar.success("✅ Model Ready")
except Exception as e:
    st.sidebar.error(f"❌ Error: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# MAIN CONTENT - NO PADDING
# -----------------------------------------------------------------------------
st.markdown('<div class="upload-section">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📤 Upload a landmark photo",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Loaded Image", use_column_width=True)

    with st.spinner("🔍 Analyzing the image..."):
        result = get_prediction(image)

    if result is None:
        st.error("❌ Could not identify landmark. Please upload a clear photo of a landmark (no humans).")
    else:
        st.markdown(f"""
        <div class="result-card">
            <h3 style="margin-top: 0;">✨ Prediction Result</h3>
            <p style="font-size: 1.2rem;"><strong>📌 Landmark:</strong><br>{result['name'].strip()}</p>
            <p style="font-size: 1.2rem;"><strong>📍 Location:</strong><br>{result['place'].strip()}</p>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #b8b8b8;">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📸</div>
        <p style="margin: 0.5rem 0; font-size: 1.2rem; opacity: 0.9">Ready to Explore?</p>
        <p style="margin: 0.5rem 0; font-size: 1.1rem; opacity: 0.9">Upload a photo of a Sri Lankan landmark above to get started!</p>
        <p style="font-size: 0.9rem; color: #FCFAFA; margin: 0.5rem 0; opacity: 0.8">
            💡 Tip: Clear, well-lit photos work best
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# FOOTER - Full Width
# -----------------------------------------------------------------------------
st.markdown("""
<div class="footer">
    <p style="margin: 0; color: #ffffff; opacity: 0.8">AI-Powered Intelligent Tourism Support System</p>
    <p style="margin: 0.3rem 0 0 0; font-size: 0.7rem; color: #ffffff; opacity: 0.8">© 2026</p>
</div>
""", unsafe_allow_html=True)