import streamlit as st
import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import monkey patch first to fix OverflowError
from src.utils.monkey_patch import monkeypatch
monkeypatch()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import TensorFlow with error handling
try:
    import tensorflow as tf
    tensorflow_available = True
except ImportError as e:
    tensorflow_available = False
    tensorflow_error = str(e)

import librosa
import librosa.display
import soundfile as sf
import time
import subprocess
import threading
import queue
from pathlib import Path
from streamlit_option_menu import option_menu
from streamlit_extras.colored_header import colored_header
from streamlit_extras.app_logo import add_logo
from streamlit_extras.card import card
from streamlit_extras.stylable_container import stylable_container
import plotly.express as px
import tempfile
import webbrowser
import socket
import plotly.graph_objects as go

# Import custom modules
from src.models.emotion_model import EmotionModel
from src.features.feature_extractor import FeatureExtractor
from src.ui.dashboard import EmotionDashboard

# Set page configuration
st.set_page_config(
    page_title="Speech Emotion Analyzer",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Status container for training updates
status_container = st.empty()

# Display training status from session state
if 'training_status' in st.session_state:
    status = st.session_state.training_status
    if status['type'] == 'progress':
        status_container.markdown(f"""
        <div class='status-container status-progress'>
            <h4>⏳ Training in progress...</h4>
            <p>{status['message']}</p>
        </div>
        """, unsafe_allow_html=True)
    elif status['type'] == 'success':
        status_container.markdown(f"""
        <div class='status-container status-success'>
            <h4>✅ Training completed!</h4>
            <p>{status['message']}</p>
        </div>
        """, unsafe_allow_html=True)
    elif status['type'] == 'error':
        status_container.markdown(f"""
        <div class='status-container status-error'>
            <h4>❌ Training failed!</h4>
            <p>{status['message']}</p>
        </div>
        """, unsafe_allow_html=True)

# Define CSS styles
st.markdown("""
<style>
    /* Modern typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Status container styling */
    .status-container {
        padding: 12px;
        border-radius: 8px;
        margin: 16px 0;
        border: 1px solid transparent;
    }
    
    .status-progress {
        background-color: #e6f7ff;
        border-color: #1890ff;
    }
    
    .status-success {
        background-color: #d4edda;
        border-color: #28a745;
    }
    
    .status-error {
        background-color: #f8d7da;
        border-color: #dc3545;
    }
    
    /* Main styling */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main header styles */
    .main-header {
        font-size: 2.75rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6A11CB 0%, #2575FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        padding-top: 1rem;
        letter-spacing: -0.5px;
    }
    
    /* Subheader styles */
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #4F46E5;
        margin-bottom: 1.25rem;
        letter-spacing: -0.3px;
    }
    
    /* Card container with modern shadow */
    .card-container {
        border-radius: 16px;
        padding: 24px;
        background-color: #FFFFFF;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 20px 25px -5px rgba(0,0,0,0.05);
        margin-bottom: 24px;
        border: 1px solid rgba(226, 232, 240, 0.8);
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    
    .card-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1);
    }
    
    /* Emotion cards */
    .emotion-card {
        background-color: white;
        border-radius: 14px;
        padding: 20px;
    }
    
    .emotion-icon {
        font-size: 3rem;
        text-align: center;
        margin-bottom: 12px;
    }
    
    .emotion-title {
        font-weight: 600;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 8px;
    }
    
    .emotion-description {
        color: #6B7280;
        font-size: 0.9rem;
        text-align: center;
        margin-bottom: 16px;
        line-height: 1.5;
    }
    
    /* Confidence meter */
    .confidence-meter {
        height: 8px;
        background-color: #E2E8F0;
        border-radius: 100px;
        margin: 16px 0;
        overflow: hidden;
    }
    
    /* Info box */
    .info-box {
        background-color: #F8FAFC;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 20px;
        border: 1px solid #E2E8F0;
    }
    
    /* Accuracy meter */
    .accuracy-meter {
        height: 8px;
        width: 100%;
        background-color: #E2E8F0;
        border-radius: 100px;
        margin-top: 10px;
        position: relative;
        overflow: hidden;
    }
    
    .accuracy-value {
        position: absolute;
        height: 100%;
        width: 80%;
        background: linear-gradient(90deg, #4F46E5 0%, #818CF8 100%);
        border-radius: 100px;
    }
    
    div[data-testid="stStatusWidget"] {
        display: none;
    }
    
    /* Upload area */
    .uploadArea div[data-testid="stFileUploader"] {
        padding: 2rem 1rem;
        background-color: #F8FAFC;
        border: 2px dashed #CBD5E1;
        border-radius: 16px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .uploadArea div[data-testid="stFileUploader"]:hover {
        border-color: #93C5FD;
        background-color: #F0F9FF;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3.5rem;
        white-space: pre-wrap;
        font-weight: 500;
        background-color: #F5F7FF;
        border-radius: 12px;
        padding: 0.75rem 1.25rem;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #EBEEFE;
    }
    
    /* Button hover effects */
    .stButton button {
        border-radius: 100px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# Define emotion colors for visualization
EMOTION_COLORS = {
    "neutral": "#607D8B",
    "calm": "#1E88E5",
    "happy": "#FFB300",
    "sad": "#5E35B1",
    "angry": "#D32F2F",
    "fearful": "#7CB342",
    "disgust": "#00897B",
    "surprised": "#F06292"
}

class EmotionAnalyzer:
    def __init__(self):
        self.upload_folder = os.path.join(os.path.abspath(os.getcwd()), "uploads")
        self.model_path = "models/emotion_model"
        self.backup_model_path = "models/emotion_model.h5"
        
        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.dashboard = EmotionDashboard()
        
        # Set default emotion labels
        self.emotion_labels = [
            "neutral", "calm", "happy", "sad", "angry", 
            "fearful", "disgust", "surprised"
        ]
        
        # Internal state
        self.loaded = False
        self.model = None
        self.tensorflow_available = tensorflow_available
        
        # Ensure upload directory exists
        self.ensure_upload_dir()

    def ensure_upload_dir(self):
        try:
            os.makedirs(self.upload_folder, mode=0o755, exist_ok=True)
        except OSError as e:
            st.error(f"Cannot create upload directory: {e}")
            raise

    def load_model(self):
        if not self.tensorflow_available:
            st.error(f"TensorFlow is not available. Error: {tensorflow_error}")
            st.stop()
        
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                st.success("✅ Model loaded successfully!")
                self.loaded = True
                return True
            else:
                st.warning("No model found. Please train a model first.")
                return False
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False

    def run(self):
        st.markdown("""
        <div class='main-header'>Speech Emotion Analyzer</div>
        <p style="text-align: center; font-size: 1.1rem; color: #6B7280;">
            Detect emotions in speech using AI-powered deep learning technology
        </p>
        """, unsafe_allow_html=True)

        if not self.loaded:
            self.load_model()

if __name__ == "__main__":
    app = EmotionAnalyzer()
    app.run()
