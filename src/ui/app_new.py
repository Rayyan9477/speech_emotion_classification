import streamlit as st
import os
import sys
import numpy as np
import logging
from pathlib import Path
import subprocess
import time
import threading

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
        self.training_in_progress = False
        self.training_process = None
        self.training_thread = None

    def load_model(self):
        """Load the model from available sources or automatically train a new one if none found"""
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

            # If no model found, start automatic training
            st.info("📊 No trained model found. Starting automatic model training...")
            if self.training_in_progress:
                st.warning("⏳ Model training is already in progress. Please wait...")
                return False
            
            # Start the training process
            success = self.train_model_automatically()
            if success:
                # Training started successfully
                return self.check_and_load_new_model()
            else:
                # Failed to start training
                st.error("❌ Failed to start model training. Please check the logs for details.")
                return False

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            st.error(f"❌ Error loading model: {str(e)}")
            return False

    def train_model_automatically(self):
        """Start the model training process automatically"""
        try:
            self.training_in_progress = True
            st.info("🚀 Starting automatic model training. This may take a few minutes...")
            
            # Display a progress message
            progress_placeholder = st.empty()
            progress_placeholder.markdown("""
            <div style='padding: 10px; border-radius: 5px; background-color: #e6f7ff; border: 1px solid #1890ff; margin: 10px 0;'>
                <h4 style='margin: 0; color: #096dd9;'>⏳ Training in progress...</h4>
                <p style='margin: 5px 0 0 0;'>Training a new emotion recognition model. This process may take 5-10 minutes depending on your system.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create and start the training process
            self.training_process = subprocess.Popen(
                [sys.executable, "-m", "src.main", "--train", "--model-type", "cnn"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Create a thread to monitor the training process
            self.training_thread = threading.Thread(
                target=self._monitor_training_process,
                args=(progress_placeholder,),
                daemon=True
            )
            self.training_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting training process: {e}")
            self.training_in_progress = False
            return False

    def _monitor_training_process(self, progress_placeholder):
        """Monitor the training process and update the UI with progress"""
        try:
            start_time = time.time()
            
            while self.training_process.poll() is None:
                elapsed = time.time() - start_time
                mins, secs = divmod(int(elapsed), 60)
                
                # Update progress message every few seconds
                progress_placeholder.markdown(f"""
                <div style='padding: 10px; border-radius: 5px; background-color: #e6f7ff; border: 1px solid #1890ff; margin: 10px 0;'>
                    <h4 style='margin: 0; color: #096dd9;'>⏳ Training in progress...</h4>
                    <p style='margin: 5px 0 0 0;'>Training a new emotion recognition model. Time elapsed: {mins:02d}:{secs:02d}</p>
                </div>
                """, unsafe_allow_html=True)
                
                time.sleep(2)
            
            # Process completed
            return_code = self.training_process.returncode
            stdout, stderr = self.training_process.communicate()
            
            if return_code == 0:
                # Training successful
                elapsed = time.time() - start_time
                mins, secs = divmod(int(elapsed), 60)
                progress_placeholder.markdown(f"""
                <div style='padding: 10px; border-radius: 5px; background-color: #d4edda; border: 1px solid #28a745; margin: 10px 0;'>
                    <h4 style='margin: 0; color: #155724;'>✅ Training completed!</h4>
                    <p style='margin: 5px 0 0 0;'>Training completed successfully in {mins:02d}:{secs:02d}. Attempting to load the new model...</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Try to load the newly trained model
                self.load_trained_model()
            else:
                # Training failed
                progress_placeholder.markdown(f"""
                <div style='padding: 10px; border-radius: 5px; background-color: #f8d7da; border: 1px solid #dc3545; margin: 10px 0;'>
                    <h4 style='margin: 0; color: #721c24;'>❌ Training failed!</h4>
                    <p style='margin: 5px 0 0 0;'>There was an error during model training. Please check the logs for details.</p>
                </div>
                """, unsafe_allow_html=True)
                logger.error(f"Training process failed with return code {return_code}")
                logger.error(f"Error output: {stderr}")
            
            # Reset training status
            self.training_in_progress = False
            
        except Exception as e:
            logger.error(f"Error in monitoring training process: {e}")
            self.training_in_progress = False

    def load_trained_model(self):
        """Try to load the newly trained model after training completes"""
        try:
            # Short delay to ensure file system has updated
            time.sleep(1)
            
            # Check if the model file exists now
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                self.loaded = True
                st.success("✅ Newly trained model loaded successfully!")
                st.experimental_rerun()
                return True
            
            # If main model not found, check in logs directory for the most recent model
            log_dir = Path('logs')
            if log_dir.exists():
                run_dirs = sorted(log_dir.glob('run_*'), reverse=True)
                if run_dirs:
                    latest_run = run_dirs[0]
                    model_path = latest_run / 'best_model.keras'
                    if model_path.exists():
                        self.model = tf.keras.models.load_model(str(model_path))
                        self.model_path = str(model_path)
                        self.loaded = True
                        st.success(f"✅ Newly trained model loaded from {latest_run.name}!")
                        st.experimental_rerun()
                        return True
            
            st.warning("⚠️ Training completed but couldn't find the new model. Please refresh the page.")
            return False
            
        except Exception as e:
            logger.error(f"Error loading trained model: {e}")
            st.error(f"❌ Error loading model after training: {str(e)}")
            return False

    def check_and_load_new_model(self):
        """Poll periodically for the new model to become available"""
        if not self.training_in_progress:
            return False
            
        # Display waiting message
        st.info("⏳ Model training in progress. You can continue using other features, and the page will automatically reload when training completes.")
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