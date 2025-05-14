# filepath: c:\Users\rayyan.a\Downloads\Repo\speech_emotion_classification\src\ui\app.py

import streamlit as st
import os
import sys
import numpy as np
import logging
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import monkey patch first to fix OverflowError
from src.utils.monkey_patch import monkeypatch
monkeypatch()

# Try to import TensorFlow with error handling
try:
    import tensorflow as tf
    tensorflow_available = True
except ImportError as e:
    tensorflow_available = False
    tensorflow_error = str(e)

import librosa
import plotly.graph_objects as go

# Import custom modules
from src.models.emotion_model import EmotionModel
from src.features.feature_extractor import FeatureExtractor
from src.models.model_manager import ModelManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmotionAnalyzer:
    def __init__(self):
        self.model = None
        self.model_manager = ModelManager()
        self.feature_extractor = FeatureExtractor()
        self.model_path = os.path.join('models', 'cnn_emotion_model.keras')
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.loaded = False
        self.emoji_map = {
            'angry': '😠',
            'disgust': '🤢',
            'fear': '😨',
            'happy': '😊',
            'neutral': '😐',
            'sad': '😢',
            'surprise': '😲'
        }

    def load_model(self):
        """Load the model from available sources"""
        try:
            # First try to load from direct file
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                st.success(f"✅ Model loaded successfully!")
                self.loaded = True
                return True

            # Check logs directory for backup models
            log_dir = Path('logs')
            if log_dir.exists():
                run_dirs = sorted(log_dir.glob('run_*'), reverse=True)
                for run_dir in run_dirs:
                    model_path = run_dir / 'best_model.keras'
                    if model_path.exists():
                        self.model = tf.keras.models.load_model(str(model_path))
                        st.success(f"✅ Model loaded successfully from backup!")
                        self.model_path = str(model_path)
                        self.loaded = True
                        return True

            st.warning("⚠️ No trained model found. Please train a model first.")
            return False

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            st.error(f"❌ Error loading model: {str(e)}")
            return False

    def process_audio(self, audio_file):
        """Process audio file and predict emotion"""
        try:
            if not self.loaded and not self.load_model():
                return

            # Load and preprocess audio
            y, sr = librosa.load(audio_file)
            
            # Display waveform
            st.subheader("🌊 Audio Waveform")
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=y, mode='lines', name='Waveform'))
            fig.update_layout(
                title="Audio Signal",
                xaxis_title="Sample",
                yaxis_title="Amplitude",
                showlegend=False,
                plot_bgcolor='white'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Extract features
            features = self.feature_extractor.extract_features(y, sr)
            if features is None:
                st.error("❌ Failed to extract features from audio!")
                return

            # Make prediction
            predictions = self.model.predict(features)
            emotion_probs = dict(zip(self.emotion_labels, predictions[0]))

            # Display results
            self.display_results(emotion_probs)

        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            st.error(f"❌ Error processing audio: {str(e)}")

    def display_results(self, emotion_probs):
        """Display emotion analysis results"""
        st.subheader("🎭 Emotion Analysis Results")

        # Create bar chart with custom colors
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99CCFF', '#FFB366']
        fig = go.Figure(data=[
            go.Bar(
                x=list(emotion_probs.keys()),
                y=list(emotion_probs.values()),
                marker_color=colors
            )
        ])

        fig.update_layout(
            title={
                'text': "Emotion Probabilities",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Emotion",
            yaxis_title="Probability",
            yaxis_range=[0, 1],
            plot_bgcolor='white'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show top 3 emotions with emojis
        st.subheader("🏆 Top 3 Detected Emotions:")
        top_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for i, (emotion, prob) in enumerate(top_emotions, 1):
            emoji = self.emoji_map.get(emotion, '')
            st.markdown(
                f"""
                <div style='
                    padding: 10px;
                    border-radius: 5px;
                    background-color: {"#FFD700" if i==1 else "#C0C0C0" if i==2 else "#CD7F32"};
                    margin: 5px 0;
                    text-align: center;
                '>
                    <h3>{i}. {emoji} {emotion.capitalize()}: {prob:.1%}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

def main():
    # Page config
    st.set_page_config(
        page_title="Speech Emotion Analyzer",
        page_icon="🎭",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            padding: 20px 0;
        }
        .stAlert {
            padding: 1rem;
            border-radius: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("🎙️ Speech Emotion Analyzer")
    st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            This application analyzes the emotional content in speech audio files.
            Upload a WAV file to get started!
        </div>
    """, unsafe_allow_html=True)

    # Initialize analyzer
    analyzer = EmotionAnalyzer()

    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📤 Upload Your Audio")
        uploaded_file = st.file_uploader(
            "Choose a WAV file",
            type=['wav'],
            help="Upload a WAV file containing speech"
        )

    with col2:
        st.subheader("🎵 Try a Demo")
        demo_files = {
            "Happy Sample": "demo_files/happy_sample.wav",
            "Sad Sample": "demo_files/sad_sample.wav",
            "Angry Sample": "demo_files/angry_sample.wav"
        }
        selected_demo = st.selectbox(
            "Select a demo file:",
            ["Select a demo..."] + list(demo_files.keys())
        )

    # Process audio
    if uploaded_file is not None:
        st.audio(uploaded_file)
        analyzer.process_audio(uploaded_file)
    elif selected_demo != "Select a demo...":
        demo_path = demo_files[selected_demo]
        if os.path.exists(demo_path):
            st.audio(demo_path)
            analyzer.process_audio(demo_path)
        else:
            st.error(f"❌ Demo file not found: {demo_path}")

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <h4>📊 Supported Emotions</h4>
            <div style='display: flex; justify-content: center; flex-wrap: wrap; gap: 10px;'>
                <span>😠 Angry</span>
                <span>🤢 Disgust</span>
                <span>😨 Fear</span>
                <span>😊 Happy</span>
                <span>😐 Neutral</span>
                <span>😢 Sad</span>
                <span>😲 Surprise</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
