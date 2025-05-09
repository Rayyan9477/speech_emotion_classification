import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import monkey patch first to fix OverflowError
import monkey_patch
monkey_patch.monkeypatch()

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

# Import the audiorecorder component
try:
    from streamlit_audiorecorder import st_audiorecorder
    audiorecorder_available = True
except ImportError:
    audiorecorder_available = False

# Import custom modules
from model import EmotionModel
from feature_extractor import FeatureExtractor
from dashboard import EmotionDashboard

# Set page configuration
st.set_page_config(
    page_title="Speech Emotion Analyzer",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI styling
st.markdown("""
<style>
    /* Modern typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
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
        transform: translateY(-3px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05), 0 20px 25px -5px rgba(0,0,0,0.09);
    }
    
    /* Result container */
    .result-container {
        background-color: #F9FAFB;
        border-radius: 16px;
        padding: 28px;
        margin-top: 24px;
        border: 1px solid rgba(226, 232, 240, 0.8);
    }
    
    /* Emotion tag with modern design */
    .emotion-tag {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 100px;
        margin-right: 8px;
        font-weight: 600;
        font-size: 0.85rem;
        color: white;
        letter-spacing: 0.3px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.08);
        transition: transform 0.15s ease;
    }
    
    .emotion-tag:hover {
        transform: translateY(-1px);
    }
    
    /* Progress bar */
    .stProgress .st-bo {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
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
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #4F46E5;
        color: white;
    }
    
    /* Gauge chart styling */
    .gauge-chart {
        margin: 0 auto;
        text-align: center;
        border-radius: 16px;
        padding: 1rem;
        background-color: #FFFFFF;
    }
    
    /* Alert animations */
    .stAlert {
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: all 0.3s ease-in-out;
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
    
    /* TensorBoard launcher */
    .tb-launcher {
        background-color: #F0F7FF;
        border-radius: 14px;
        padding: 20px;
        border: 1px solid #D1E9FF;
        margin-top: 16px;
    }
    
    /* Real-time indicator with modern animation */
    .real-time-indicator {
        color: #22C55E;
        font-weight: 600;
        font-size: 0.9rem;
        animation: modern-pulse 1.5s infinite;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    .real-time-indicator:before {
        content: "";
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #22C55E;
    }
    
    @keyframes modern-pulse {
        0% {
            opacity: 0.7;
        }
        50% {
            opacity: 1;
        }
        100% {
            opacity: 0.7;
        }
    }
    
    /* Improved file upload area */
    .uploadArea {
        border: 2px dashed #E2E8F0;
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        background-color: #F8FAFC;
        transition: all 0.2s ease;
    }
    
    .uploadArea:hover {
        border-color: #4F46E5;
        background-color: #F5F7FF;
    }
    
    /* Audio controls styling */
    audio {
        width: 100%;
        border-radius: 100px;
        height: 40px;
        background-color: #F5F7FF;
    }
    
    /* Sidebar styling */
    .css-1vq4p4l {
        padding: 1.5rem 1rem !important;
    }
    
    /* Mobile responsiveness improvements */
    @media screen and (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .sub-header {
            font-size: 1.25rem;
        }
        
        .card-container, .result-container {
            padding: 18px;
        }
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

EMOTION_DESCRIPTIONS = {
    "neutral": "A balanced emotional state without strong positive or negative feeling.",
    "calm": "A peaceful, relaxed state free from agitation or excitement.",
    "happy": "A state of joy, pleasure, or contentment.",
    "sad": "A state of sorrow, unhappiness, or melancholy.",
    "angry": "A strong feeling of displeasure, hostility, or antagonism.",
    "fearful": "A state of being afraid or anxious about something threatening or dangerous.",
    "disgust": "A feeling of revulsion or strong disapproval.",
    "surprised": "A feeling of being startled or astonished by something unexpected."
}

# Global queues for real-time processing
audio_queue = queue.Queue()
result_queue = queue.Queue()

# Check if port is available
def is_port_available(port):
    """Check if a port is available for use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

# Find an available port for TensorBoard
def find_available_port(start_port=6006):
    """Find an available port starting from the given port"""
    port = start_port
    while not is_port_available(port):
        port += 1
    return port

# Start TensorBoard process
def start_tensorboard(logdir, port=6006):
    """Start TensorBoard server in a new process"""
    try:
        # Kill any existing TensorBoard processes
        if os.name == 'nt':  # Windows
            try:
                subprocess.run(["taskkill", "/f", "/im", "tensorboard.exe"], 
                            stdout=subprocess.DEVNULL, 
                            stderr=subprocess.DEVNULL)
            except:
                pass
        else:  # Linux/Mac
            try:
                subprocess.run(["pkill", "-f", "tensorboard"], 
                            stdout=subprocess.DEVNULL, 
                            stderr=subprocess.DEVNULL)
            except:
                pass
        
        # Start new TensorBoard process
        try:
            # Check if logdir exists and has content
            if not os.path.exists(logdir):
                raise FileNotFoundError(f"Log directory {logdir} does not exist")
            
            # Check if the directory has any content
            if not any(os.scandir(logdir)):
                raise ValueError(f"Log directory {logdir} is empty, no training logs found")
                
            # Launch TensorBoard process
            process = subprocess.Popen(
                ["tensorboard", "--logdir", logdir, "--port", str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Check if process started successfully
            if process.poll() is not None:
                # Process terminated immediately
                stdout, stderr = process.communicate()
                if stderr:
                    raise RuntimeError(f"TensorBoard failed to start: {stderr}")
            
            return process, port
        except FileNotFoundError as e:
            st.error(f"Error: {str(e)}")
            return None, None
        except ValueError as e:
            st.error(f"Error: {str(e)}")
            return None, None
        except Exception as e:
            st.error(f"Failed to start TensorBoard: {e}")
            return None, None
    except Exception as e:
        st.error(f"Unexpected error starting TensorBoard: {e}")
        return None, None

# Real-time audio processing thread
def process_audio_thread(model, feature_extractor, emotion_labels):
    """Background thread for processing audio in real-time"""
    while True:
        audio_data = audio_queue.get()
        if audio_data is None:  # Signal to stop the thread
            break
            
        try:
            # Process the audio
            y, sr = audio_data
            
            # Extract features
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128, fmax=8000
            )
            
            # Convert to decibels
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Expected shape for the model input is (1, 128, 165, 1)
            expected_shape = (1, 128, 165, 1)
            
            # Fix time dimension (axis 1 in mel_spec_db)
            time_frames = mel_spec_db.shape[1]
            expected_frames = expected_shape[2]  # 165 frames
            
            if time_frames < expected_frames:
                # Pad if shorter
                padding = ((0, 0), (0, expected_frames - time_frames))
                mel_spec_db = np.pad(mel_spec_db, padding, mode='constant')
            elif time_frames > expected_frames:
                # Trim if longer
                mel_spec_db = mel_spec_db[:, :expected_frames]
            
            # Reshape for model input (adding batch and channel dimensions)
            mel_spec_db = mel_spec_db.reshape(1, mel_spec_db.shape[0], mel_spec_db.shape[1], 1)
            
            # Double-check the shape matches what the model expects
            actual_shape = mel_spec_db.shape
            if actual_shape != expected_shape:
                print(f"Feature shape mismatch: Expected {expected_shape}, got {actual_shape}. Fixing...")
                
                # Create a new array with the correct shape
                fixed_features = np.zeros(expected_shape)
                
                # Copy data, trimming or padding as needed for each dimension
                # Handle batch dimension (usually 1)
                min_batch = min(actual_shape[0], expected_shape[0])
                # Handle frequency dimension (128 mel bands)
                min_freq = min(actual_shape[1], expected_shape[1])
                # Handle time dimension (165 frames)
                min_time = min(actual_shape[2], expected_shape[2])
                # Handle channel dimension (usually 1)
                min_channel = min(actual_shape[3], expected_shape[3])
                
                # Copy only what fits
                fixed_features[:min_batch, :min_freq, :min_time, :min_channel] = \
                    mel_spec_db[:min_batch, :min_freq, :min_time, :min_channel]
                
                mel_spec_db = fixed_features
            
            # Make prediction
            prediction = model.predict(mel_spec_db, verbose=0)
            predicted_class = np.argmax(prediction[0])
            emotion = emotion_labels[predicted_class] if predicted_class < len(emotion_labels) else "unknown"
            
            # Get confidence scores
            confidence_scores = {}
            for i, label in enumerate(emotion_labels[:len(prediction[0])]):
                confidence_scores[label] = float(prediction[0][i]) * 100
                
            # Put results in the queue
            result_queue.put((emotion, confidence_scores, y, sr))
            
        except Exception as e:
            print(f"Error in audio processing thread: {e}")
            import traceback
            print(traceback.format_exc())
        
        finally:
            audio_queue.task_done()

class EmotionAnalyzer:
    """Main class for the Emotion Analysis Application"""
    
    def __init__(self):
        """Initialize the Emotion Analyzer application"""
        # Setup paths
        self.upload_folder = "uploads"
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
        self.processing_thread = None
        self.real_time_processing = False
        self.last_prediction = None
        self.tensorboard_process = None
        self.tensorboard_port = 6006
        
        # Check tensorflow availability
        self.tensorflow_available = tensorflow_available
        
        # Ensure upload directory exists
        self.ensure_upload_dir()
        
    def ensure_upload_dir(self):
        """Ensure the upload directory exists"""
        os.makedirs(self.upload_folder, exist_ok=True)
    
    def load_model(self):
        """Load the pre-trained emotion model"""
        if not self.tensorflow_available:
            st.error(f"TensorFlow is not available. Error: {tensorflow_error}")
            st.info("Please reinstall TensorFlow or fix the DLL loading issue to use this application.")
            st.stop()
            
        try:
            with st.spinner("Loading model... Please wait."):
                # First try to load the primary model (Keras format)
                try:
                    self.model = tf.keras.models.load_model(self.model_path)
                    st.success("Model loaded successfully!")
                except Exception as e:
                    st.warning(f"Failed to load model from {self.model_path}. Error: {str(e)}")
                    st.warning("Trying backup model path...")
                    
                    # Try loading the backup model (H5 format)
                    try:
                        self.model = tf.keras.models.load_model(self.backup_model_path)
                        st.success("Backup model loaded successfully!")
                    except Exception as e2:
                        st.error(f"Failed to load backup model: {str(e2)}")
                        
                        # Try to create a fresh model with default architecture
                        try:
                            st.warning("Attempting to create a new model with default architecture...")
                            emotion_model = EmotionModel(num_classes=len(self.emotion_labels))
                            self.model = emotion_model.build_cnn(input_shape=(128, 165, 1))
                            st.warning("Created new model with default architecture. Note: This model is untrained.")
                        except Exception as e3:
                            st.error(f"Could not create default model: {str(e3)}")
                            st.error("Unable to load or create a model. Please check model files or reinstall TensorFlow.")
                            st.stop()
            
            self.loaded = True
            
            # Start the real-time processing thread if not already running
            if self.processing_thread is None or not self.processing_thread.is_alive():
                self.processing_thread = threading.Thread(
                    target=process_audio_thread, 
                    args=(self.model, self.feature_extractor, self.emotion_labels),
                    daemon=True
                )
                self.processing_thread.start()
                
        except Exception as e:
            st.error(f"Error loading model: {e}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")
            st.stop()
    
    def extract_features(self, audio_file_path):
        """Extract features from audio file for model prediction"""
        try:
            with st.spinner("Extracting audio features..."):
                # Load audio file
                y, sr = librosa.load(audio_file_path, sr=None)
                
                # Ensure consistent length (5 seconds)
                target_length = 5 * sr
                if len(y) < target_length:
                    # If audio is shorter than 5 seconds, pad with zeros
                    y = np.pad(y, (0, target_length - len(y)))
                else:
                    # If longer, trim to 5 seconds
                    y = y[:target_length]
                
                # Extract mel spectrogram with fixed parameters to match model input shape
                mel_spec = librosa.feature.melspectrogram(
                    y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128, fmax=8000
                )
                
                # Convert to decibels
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                # Expected shape for the model input is (1, 128, 165, 1)
                expected_shape = (1, 128, 165, 1)
                
                # Fix time dimension (axis 1 in mel_spec_db)
                time_frames = mel_spec_db.shape[1]
                expected_frames = expected_shape[2]  # 165 frames
                
                if time_frames < expected_frames:
                    # Pad if shorter
                    padding = ((0, 0), (0, expected_frames - time_frames))
                    mel_spec_db = np.pad(mel_spec_db, padding, mode='constant')
                elif time_frames > expected_frames:
                    # Trim if longer
                    mel_spec_db = mel_spec_db[:, :expected_frames]
                
                # Reshape for model input (adding batch and channel dimensions)
                mel_spec_db = mel_spec_db.reshape(1, mel_spec_db.shape[0], mel_spec_db.shape[1], 1)
                
                # Double-check the shape matches what the model expects
                actual_shape = mel_spec_db.shape
                if actual_shape != expected_shape:
                    st.warning(f"Feature shape mismatch: Expected {expected_shape}, got {actual_shape}. Attempting to fix...")
                    
                    # Create a new array with the correct shape
                    fixed_features = np.zeros(expected_shape)
                    
                    # Copy data, trimming or padding as needed for each dimension
                    # Handle batch dimension (usually 1)
                    min_batch = min(actual_shape[0], expected_shape[0])
                    # Handle frequency dimension (128 mel bands)
                    min_freq = min(actual_shape[1], expected_shape[1])
                    # Handle time dimension (165 frames)
                    min_time = min(actual_shape[2], expected_shape[2])
                    # Handle channel dimension (usually 1)
                    min_channel = min(actual_shape[3], expected_shape[3])
                    
                    # Copy only what fits
                    fixed_features[:min_batch, :min_freq, :min_time, :min_channel] = \
                        mel_spec_db[:min_batch, :min_freq, :min_time, :min_channel]
                    
                    mel_spec_db = fixed_features
                    st.success(f"Fixed feature shape to {mel_spec_db.shape}")
                
                return y, sr, mel_spec_db
        except Exception as e:
            st.error(f"Error extracting features: {e}")
            import traceback
            st.error(f"Details: {traceback.format_exc()}")
            return None, None, None
    
    def predict_emotion(self, features):
        """Predict emotion from audio features"""
        if not self.loaded:
            self.load_model()
            
        try:
            with st.spinner("Predicting emotion..."):
                # Make prediction
                prediction = self.model.predict(features, verbose=0)
                predicted_class = np.argmax(prediction[0])
                emotion = self.emotion_labels[predicted_class] if predicted_class < len(self.emotion_labels) else "unknown"
                
                # Get confidence scores for all emotions
                confidence_scores = {}
                for i, label in enumerate(self.emotion_labels[:len(prediction[0])]):
                    confidence_scores[label] = float(prediction[0][i]) * 100
                
                return emotion, confidence_scores
        except Exception as e:
            st.error(f"Error predicting emotion: {e}")
            return "unknown", {}
    
    def visualize_audio(self, y, sr):
        """Create visualizations of the audio signal"""
        try:
            fig, ax = plt.subplots(2, 1, figsize=(10, 6))
            
            # Time domain visualization
            librosa.display.waveshow(y, sr=sr, ax=ax[0])
            ax[0].set_title("Audio Waveform")
            ax[0].set_ylabel("Amplitude")
            
            # Frequency domain visualization (mel spectrogram)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
            S_dB = librosa.power_to_db(mel_spec, ref=np.max)
            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax[1])
            ax[1].set_title("Mel Spectrogram")
            fig.colorbar(img, ax=ax[1], format='%+2.0f dB')
            
            plt.tight_layout()
            return fig
        except Exception as e:
            st.error(f"Error visualizing audio: {e}")
            return None
    
    def create_interactive_visualization(self, confidence_scores):
        """Create interactive visualization of prediction results"""
        try:
            df = pd.DataFrame({
                'Emotion': list(confidence_scores.keys()),
                'Confidence (%)': list(confidence_scores.values())
            })
            
            # Create color mapping based on EMOTION_COLORS
            colors = [EMOTION_COLORS.get(emotion, "#607D8B") for emotion in df['Emotion']]
            
            fig = px.bar(
                df, x='Emotion', y='Confidence (%)',
                title='Emotion Prediction Confidence',
                color='Emotion',
                color_discrete_map={emotion: EMOTION_COLORS.get(emotion, "#607D8B") for emotion in df['Emotion']},
                template='plotly_white'
            )
            
            fig.update_layout(
                font=dict(size=16),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=20),
                height=400
            )
            
            return fig
        except Exception as e:
            st.error(f"Error creating visualization: {e}")
            return None
    
    def create_gauge_chart(self, confidence, emotion):
        """Create a modern gauge chart to show confidence level"""
        try:
            color = EMOTION_COLORS.get(emotion, "#607D8B")
            
            # Parse hex color into RGB components for gradient effects
            try:
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
                
                # Create color variants with transparency
                light_variant = f"rgba({r}, {g}, {b}, 0.2)"
                medium_variant = f"rgba({r}, {g}, {b}, 0.5)"
                color_rgba = f"rgba({r}, {g}, {b}, 0.9)"
            except:
                # If hex parsing fails, use fallback colors
                light_variant = "rgba(200, 200, 200, 0.2)"
                medium_variant = "rgba(150, 150, 150, 0.5)"
                color_rgba = color
            
            # Define confidence level category
            confidence_level = ""
            if confidence < 30:
                confidence_level = "Low"
            elif confidence < 70:
                confidence_level = "Moderate"
            else:
                confidence_level = "High"
                
            # Create modern gauge with smoother gradients and cleaner design
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=confidence,
                domain={"x": [0, 1], "y": [0, 1]},
                delta={"reference": 50, "increasing": {"color": color}},
                title={
                    "text": f"<b>{emotion.capitalize()}</b><br><span style='font-size:0.8em;color:gray'>{confidence_level} Confidence</span>", 
                    "font": {"size": 22, "color": color, "family": "Inter, Arial, sans-serif"}
                },
                gauge={
                    "axis": {
                        "range": [0, 100], 
                        "tickwidth": 1, 
                        "tickcolor": "#F3F4F6",
                        "tickfont": {"size": 10, "color": "#6B7280"},
                    },
                    "bar": {
                        "color": color,
                        "thickness": 0.6,
                    },
                    "bgcolor": "white",
                    "borderwidth": 0,
                    "bordercolor": "white",
                    "steps": [
                        {"range": [0, 30], "color": "#F3F4F6"},
                        {"range": [30, 70], "color": light_variant},
                        {"range": [70, 100], "color": medium_variant}
                    ],
                    "threshold": {
                        "line": {"color": "rgba(220, 50, 50, 0.8)", "width": 2},
                        "thickness": 0.75,
                        "value": 90
                    }
                }
            ))
            
            # Update layout for cleaner modern look
            fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter, Arial, sans-serif"},
                annotations=[
                    dict(
                        x=0.5,
                        y=-0.1,
                        text=f"Confidence score shows how certain the AI is about this emotion",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        font=dict(size=11, color="#6B7280")
                    )
                ]
            )
            
            # Add custom styling touches
            fig.add_shape(
                type="rect",
                xref="paper", yref="paper",
                x0=0, y0=0,
                x1=1, y1=1,
                line=dict(width=0),
                fillcolor="rgba(255,255,255,0)",
                layer="below"
            )
            
            return fig
        except Exception as e:
            print(f"Error creating gauge chart: {e}")
            import traceback
            print(traceback.format_exc())
            
            # Return a simple fallback figure in case of error
            fallback_fig = go.Figure()
            fallback_fig.add_annotation(text="Chart creation failed", showarrow=False)
            fallback_fig.update_layout(height=250)
            return fallback_fig
            
    def display_file_upload(self):
        """Display file upload interface and handle uploaded files"""
        with st.container():
            colored_header(
                label="Analyze Your Speech",
                description="Upload or record audio to detect emotions",
                color_name="violet-70"
            )
            
            tab1, tab2 = st.tabs(["📁 Upload Audio File", "🎙️ Record Audio"])
            
            # Initialize audio_recording variable to None at the start
            audio_recording = None
            
            with tab1:
                with stylable_container(
                    key="upload_container",
                    css_styles="""
                        {
                            background-color: #FFFFFF;
                            border-radius: 16px;
                            padding: 28px;
                            margin-top: 16px;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 20px 25px -5px rgba(0,0,0,0.05);
                            border: 1px solid rgba(226, 232, 240, 0.8);
                        }
                    """
                ):
                    st.markdown("<h3 style='font-weight: 600; color: #4F46E5;'>Upload Audio File</h3>", unsafe_allow_html=True)
                    
                    # Create a modern upload area
                    st.markdown("<div class='uploadArea'>", unsafe_allow_html=True)
                    uploaded_file = st.file_uploader(
                        "Choose an audio file (WAV or MP3)", 
                        type=["wav", "mp3"],
                        help="Upload a short audio clip (ideally 5-10 seconds) of someone speaking"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Tips section with modern styling
                    tips_col1, tips_col2 = st.columns([1, 1])
                    
                    with tips_col1:
                        st.markdown("""
                        <div style="background-color: #F0F7FF; border-radius: 12px; padding: 16px; height: 100%;">
                            <h4 style="color: #4F46E5; margin-bottom: 12px; font-weight: 600; font-size: 1rem;">Tips for Best Results</h4>
                            <ul style="margin-left: 0; padding-left: 20px;">
                                <li>Use clear audio with minimal background noise</li>
                                <li>Short clips (5-10 seconds) work best</li>
                                <li>Make sure the speaker's voice is prominent</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with tips_col2:
                        st.markdown("""
                        <div style="background-color: #EBEEFE; border-radius: 12px; padding: 16px; height: 100%;">
                            <h4 style="color: #4F46E5; margin-bottom: 12px; font-weight: 600; font-size: 1rem;">Supported Files</h4>
                            <p>WAV or MP3 audio files containing clear speech</p>
                            <p style="margin-top: 8px; font-size: 0.9rem;">Maximum recommended length: 10 seconds</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Sample button with improved styling
                    st.markdown("<div style='margin-top: 20px; text-align: center;'>", unsafe_allow_html=True)
                    if st.button("🔊 Try a Sample Audio", key="try_sample", use_container_width=True):
                        if os.path.exists("demo_files/happy_sample.wav"):
                            self.process_audio("demo_files/happy_sample.wav")
                            st.success("Sample audio loaded and analyzed!")
                        else:
                            st.warning("Demo files not found. Please go to the 'View Examples' section.")
                    st.markdown("</div>", unsafe_allow_html=True)
            
            with tab2:
                with stylable_container(
                    key="record_container",
                    css_styles="""
                        {
                            background-color: #FFFFFF;
                            border-radius: 16px;
                            padding: 28px;
                            margin-top: 16px;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 20px 25px -5px rgba(0,0,0,0.05);
                            border: 1px solid rgba(226, 232, 240, 0.8);
                        }
                    """
                ):
                    st.markdown("<h3 style='font-weight: 600; color: #4F46E5;'>Record Your Voice</h3>", unsafe_allow_html=True)
                    
                    # Create 2 columns for recorder and settings
                    recorder_col, settings_col = st.columns([3, 2])
                    
                    with recorder_col:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #F5F7FF 0%, #EDF2FF 100%); 
                                    border-radius: 16px; 
                                    padding: 24px; 
                                    text-align: center;
                                    border: 1px solid #E5EDFF;">
                            <h4 style="margin-bottom: 16px; color: #4338CA; font-weight: 600;">Voice Recorder</h4>
                            <p style="margin-bottom: 20px; font-size: 0.95rem;">
                                Speak clearly for 5-10 seconds to get the most accurate emotion analysis
                            </p>
                        """, unsafe_allow_html=True)
                        
                        # Check if audio recorder component is available
                        if not audiorecorder_available:
                            st.error("Audio recorder component is not available. Please use the Upload option instead.")
                            st.info("If you want to use the recording feature, run: 'pip install streamlit-audiorecorder'")
                        else:
                            # Use the st_audiorecorder component
                            audio_recording = st_audiorecorder(
                                text="Click to Record",
                                recording_color="#6366F1",
                                neutral_color="#4F46E5",
                                sample_rate=16000,
                            )
                            
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with settings_col:
                        st.markdown("<h4 style='color: #4338CA; font-weight: 600; font-size: 1.1rem; margin-bottom: 16px;'>Settings</h4>", unsafe_allow_html=True)
                        
                        # Add a toggle for real-time processing with improved styling
                        real_time_enabled = st.toggle("Real-time Analysis", value=False, 
                                              help="Process audio as you speak for immediate feedback")
                        
                        if real_time_enabled:
                            st.markdown("<p class='real-time-indicator'>Real-time analysis active</p>", unsafe_allow_html=True)
                            if not self.loaded:
                                self.load_model()
                            self.real_time_processing = True
                        else:
                            self.real_time_processing = False
                        
                        # Add recording tips
                        st.markdown("""
                        <div style="background-color: #F0FDF4; 
                                    border-radius: 12px; 
                                    padding: 12px; 
                                    margin-top: 16px;
                                    border: 1px solid #DCFCE7;">
                            <h5 style="color: #16A34A; font-size: 0.9rem; font-weight: 600; margin-bottom: 8px;">Recording Tips</h5>
                            <ul style="margin-left: 0; padding-left: 16px; font-size: 0.85rem;">
                                <li>Use a good microphone</li>
                                <li>Reduce background noise</li>
                                <li>Speak naturally</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # If real-time enabled, create a placeholder for live results
                    if real_time_enabled and audiorecorder_available:
                        st.markdown("<div style='margin-top: 20px; background-color: #F5F7FF; border-radius: 16px; padding: 20px; border: 1px dashed #C7D2FE;'>", unsafe_allow_html=True)
                        live_result_placeholder = st.empty()
                        
                        # Check if there are any results in the queue
                        if not result_queue.empty():
                            emotion, confidence_scores, audio_y, audio_sr = result_queue.get()
                            with live_result_placeholder.container():
                                st.markdown(f"<h3 style='text-align: center; font-weight: 600; color: {EMOTION_COLORS.get(emotion, '#607D8B')};'>Live Emotion: {emotion.upper()}</h3>", unsafe_allow_html=True)
                                max_confidence = confidence_scores.get(emotion, 0)
                                gauge_fig = self.create_gauge_chart(max_confidence, emotion)
                                st.plotly_chart(gauge_fig, use_container_width=True)
                        else:
                            live_result_placeholder.markdown("<p style='text-align: center; color: #6B7280;'>Start speaking to see real-time emotion analysis</p>", unsafe_allow_html=True)
                            
                        st.markdown("</div>", unsafe_allow_html=True)
            
            # Add a separator before results
            st.markdown("---")
            
            # Process the uploaded file or recorded audio
            if uploaded_file is not None:
                # Save the uploaded file to disk
                file_path = os.path.join(self.upload_folder, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.success(f"File uploaded successfully: {uploaded_file.name}")
                self.process_audio(file_path)
                
            # Check both audiorecorder_available and audio_recording to prevent errors
            elif audiorecorder_available and audio_recording is not None and len(audio_recording) > 0:
                # Save the recorded audio to disk
                timestamp = int(time.time())
                file_path = os.path.join(self.upload_folder, f"recording_{timestamp}.wav")
                with open(file_path, "wb") as f:
                    f.write(audio_recording)
                
                st.success("Recording saved successfully!")
                
                # If real-time processing is enabled, add to the queue
                if self.real_time_processing:
                    try:
                        y, sr = librosa.load(file_path, sr=None)
                        audio_queue.put((y, sr))
                    except Exception as e:
                        st.error(f"Error processing audio for real-time analysis: {e}")
                
                self.process_audio(file_path)
    
    def display_demo_section(self):
        """Display demo section with example audio files"""
        with st.container():
            colored_header(
                label="Try with Examples",
                description="Select an example audio file to analyze",
                color_name="violet-70"
            )
            
            # Create a folder for demo files if it doesn't exist
            demo_folder = "demo_files"
            os.makedirs(demo_folder, exist_ok=True)
            
            # Create a modern card layout for examples
            st.markdown("""
            <style>
                .emotion-card {
                    background-color: white;
                    border-radius: 16px;
                    padding: 24px;
                    transition: all 0.3s ease;
                    height: 100%;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 20px 25px -5px rgba(0,0,0,0.05);
                    border: 1px solid rgba(226, 232, 240, 0.8);
                    display: flex;
                    flex-direction: column;
                }
                .emotion-card:hover {
                    transform: translateY(-4px);
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                }
                .emotion-icon {
                    font-size: 2.5rem;
                    margin-bottom: 16px;
                    text-align: center;
                }
                .emotion-title {
                    font-size: 1.25rem;
                    font-weight: 600;
                    margin-bottom: 8px;
                    text-align: center;
                }
                .emotion-description {
                    font-size: 0.9rem;
                    color: #6B7280;
                    margin-bottom: 16px;
                    text-align: center;
                    flex-grow: 1;
                }
            </style>
            """, unsafe_allow_html=True)
            
            # Create a modern gallery of example cards
            col1, col2, col3 = st.columns(3)
            
            emotion_samples = [
                {
                    "emotion": "happy",
                    "icon": "😄",
                    "color": "#FFB300",
                    "gradient": "linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%)",
                    "description": "Example of a joyful, enthusiastic voice with positive tonality."
                },
                {
                    "emotion": "angry",
                    "icon": "😡",
                    "color": "#D32F2F",
                    "gradient": "linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%)",
                    "description": "Example of an irritated voice with aggressive tonality and higher intensity."
                },
                {
                    "emotion": "sad",
                    "icon": "😢",
                    "color": "#5E35B1",
                    "gradient": "linear-gradient(135deg, #EDE9FE 0%, #DDD6FE 100%)",
                    "description": "Example of a melancholic voice with lower energy and somber tone."
                }
            ]
            
            columns = [col1, col2, col3]
            
            for i, (col, sample) in enumerate(zip(columns, emotion_samples)):
                with col:
                    emotion = sample["emotion"]
                    st.markdown(f"""
                    <div style="background: {sample['gradient']}; border-radius: 16px; padding: 2px; margin-bottom: 16px;">
                        <div class="emotion-card">
                            <div class="emotion-icon">{sample["icon"]}</div>
                            <div class="emotion-title" style="color: {sample['color']};">{emotion.capitalize()} Voice</div>
                            <div class="emotion-description">{sample["description"]}</div>
                            <div style="text-align: center;">
                                <audio style="width: 100%; border-radius: 100px; height: 40px;" 
                                       src="demo_files/{emotion}_sample.wav" 
                                       controls></audio>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add analyze button with matching color
                    if st.button(f"Analyze {emotion.capitalize()}", key=f"{emotion}_btn", use_container_width=True):
                        sample_path = f"demo_files/{emotion}_sample.wav"
                        if os.path.exists(sample_path):
                            self.process_audio(sample_path)
                        else:
                            st.warning(f"Demo file for {emotion} not found. Please run the setup script first.")
            
            # Add information about the demo samples
            with st.expander("About these samples"):
                st.markdown("""
                ### About the Demo Samples
                
                These audio samples are curated examples from the RAVDESS dataset (Ryerson Audio-Visual Database of Emotional Speech and Song),
                which is commonly used in emotion recognition research.
                
                Each sample demonstrates a different emotional state expressed through voice:
                
                - **Happy**: Characterized by higher pitch, faster tempo, and energetic delivery
                - **Angry**: Features strong intensity, sharp attacks, and tense vocal qualities
                - **Sad**: Exhibits lower energy, slower pace, and descending pitch patterns
                
                You can use these samples to test the emotion recognition system or as a reference point for your own recordings.
                """)
                
            # Add a quick guide on how to interpret the results
            st.markdown("""
            <div style="background-color: #F0F9FF; border: 1px solid #BAE6FD; border-radius: 16px; padding: 20px; margin-top: 24px;">
                <h4 style="color: #0369A1; font-weight: 600; margin-bottom: 12px;">How to use the examples</h4>
                <ol style="margin-bottom: 0; padding-left: 20px;">
                    <li>Listen to the audio sample to hear the emotional expression</li>
                    <li>Click "Analyze" to process the sample with our AI</li>
                    <li>Compare the detected emotion with your own perception</li>
                    <li>Explore the confidence scores to understand the AI's decision process</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
    
    def process_audio(self, audio_file_path):
        """Process audio file and display results"""
        if not self.loaded:
            self.load_model()
        
        # Check if file exists
        if not os.path.exists(audio_file_path):
            st.error(f"Audio file not found: {audio_file_path}")
            return
            
        try:
            # Extract features from audio
            y, sr, features = self.extract_features(audio_file_path)
            
            if features is None:
                st.error("Failed to extract features from the audio file.")
                return
            
            # Predict emotion
            emotion, confidence_scores = self.predict_emotion(features)
            
            # Display results
            self.display_results(audio_file_path, y, sr, emotion, confidence_scores)
            
            # Save analysis results for dashboard visualization
            self.dashboard.save_analysis_result(audio_file_path, emotion, confidence_scores)
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
            import traceback
            st.error(f"Details: {traceback.format_exc()}")
    
    def display_results(self, audio_file_path, y, sr, emotion, confidence_scores):
        """Display emotion analysis results"""
        with st.container():
            colored_header(
                label="Analysis Results",
                description="Detected emotion and confidence scores",
                color_name="violet-70"
            )
            
            # Create a modern card for the results
            st.markdown(f"""
            <div style="background: linear-gradient(to right, {EMOTION_COLORS.get(emotion, '#607D8B') + '22'}, #FFFFFF);
                        border-radius: 20px; 
                        padding: 2px; 
                        margin-bottom: 24px;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                        border: 1px solid {EMOTION_COLORS.get(emotion, '#607D8B') + '40'}">
                <div style="background-color: white; border-radius: 18px; padding: 24px;">
                    <h2 style="text-align: center; 
                              font-weight: 700; 
                              font-size: 2rem;
                              background: linear-gradient(90deg, {EMOTION_COLORS.get(emotion, '#607D8B')}, #4F46E5);
                              -webkit-background-clip: text;
                              -webkit-text-fill-color: transparent;
                              margin-bottom: 16px;">
                        {emotion.upper()}
                    </h2>
                    <p style="text-align: center; 
                             font-size: 1.1rem; 
                             color: #4B5563; 
                             font-style: italic;
                             margin-bottom: 24px;">
                        {EMOTION_DESCRIPTIONS.get(emotion, '')}
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Create three columns for results presentation
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Confidence gauge in a modern card
                with stylable_container(
                    key="result_container",
                    css_styles="""
                        {
                            background-color: #FFFFFF;
                            border-radius: 16px;
                            padding: 24px;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 20px 25px -5px rgba(0,0,0,0.05);
                            height: 100%;
                            border: 1px solid rgba(226, 232, 240, 0.8);
                        }
                    """
                ):
                    st.markdown("<h3 style='font-weight: 600; font-size: 1.25rem; color: #4F46E5; margin-bottom: 16px;'>Confidence Level</h3>", unsafe_allow_html=True)
                    
                    # Check if confidence_scores is empty, and handle it gracefully
                    if not confidence_scores:
                        st.warning("No confidence scores available. The model may not have produced valid predictions.")
                        max_confidence = 0
                    else:
                        # Display most confident emotion with gauge chart
                        max_confidence = confidence_scores.get(emotion, 0)
                    
                    try:
                        gauge_fig = self.create_gauge_chart(max_confidence, emotion)
                        if gauge_fig is not None:
                            st.plotly_chart(gauge_fig, use_container_width=True)
                        else:
                            st.warning("Could not create confidence gauge chart.")
                    except Exception as e:
                        st.error(f"Error displaying gauge chart: {e}")
                    
                    # Display original audio for playback with modern styling
                    try:
                        st.markdown("<h4 style='font-weight: 600; font-size: 1rem; color: #4F46E5; margin-top: 16px; margin-bottom: 8px;'>Audio Recording</h4>", unsafe_allow_html=True)
                        st.audio(audio_file_path)
                    except Exception as e:
                        st.error(f"Error playing audio: {e}")
            
            with col2:
                # Use tabs for organized display of additional results
                tabs = st.tabs(["📊 Confidence Breakdown", "📈 Audio Analysis"])
                
                with tabs[0]:
                    # Check if confidence_scores is empty
                    if not confidence_scores:
                        st.warning("No confidence scores available to display.")
                    else:
                        try:
                            # Display interactive bar chart of confidence scores
                            fig = self.create_interactive_visualization(confidence_scores)
                            if fig is not None:
                                # Update the chart for a more modern look
                                fig.update_layout(
                                    template="plotly_white",
                                    font=dict(family="Inter, sans-serif", size=14),
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    margin=dict(l=20, r=20, t=30, b=20),
                                    height=350,
                                    xaxis=dict(
                                        tickfont=dict(size=12),
                                        title_font=dict(size=14, color="#4F46E5")
                                    ),
                                    yaxis=dict(
                                        tickfont=dict(size=12),
                                        title_font=dict(size=14, color="#4F46E5"),
                                        gridcolor='#F3F4F6'
                                    ),
                                    bargap=0.3,
                                )
                                # Add a light shadow to the bars
                                for i in range(len(fig.data)):
                                    fig.data[i].marker.line.width = 1
                                    fig.data[i].marker.line.color = "white"

                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show top 3 emotions as text
                                if len(confidence_scores) >= 3:
                                    top_emotions = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                                    st.markdown("#### Top 3 Emotions")
                                    for i, (emotion_name, score) in enumerate(top_emotions):
                                        st.markdown(f"""
                                        <div style="display: flex; align-items: center; margin-bottom: 8px;">
                                            <div style="background-color: {EMOTION_COLORS.get(emotion_name, '#607D8B')}; 
                                                        width: 16px; 
                                                        height: 16px; 
                                                        border-radius: 50%; 
                                                        margin-right: 8px;"></div>
                                            <div style="font-weight: 500;">{emotion_name.capitalize()}: {score:.1f}%</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                            else:
                                st.warning("Could not create confidence visualization.")
                        except Exception as e:
                            st.error(f"Error displaying confidence scores visualization: {e}")
                
                with tabs[1]:
                    try:
                        # Display audio visualizations if y and sr are available
                        if y is not None and sr is not None:
                            # Create a more modern visualization of the audio
                            fig = plt.figure(figsize=(10, 6), facecolor='#FFFFFF')
                            plt.subplots_adjust(hspace=0.5)
                            
                            # Time domain visualization with improved styling
                            ax1 = plt.subplot(2, 1, 1)
                            librosa.display.waveshow(y, sr=sr, ax=ax1, color='#4F46E5', alpha=0.7)
                            ax1.set_title("Audio Waveform", fontsize=14, fontweight='bold', color='#1F2937')
                            ax1.set_ylabel("Amplitude", fontsize=12, color='#4B5563')
                            ax1.set_facecolor('#F9FAFB')
                            ax1.grid(True, linestyle='--', alpha=0.7, color='#E5E7EB')
                            
                            # Frequency domain visualization (mel spectrogram) with improved styling
                            ax2 = plt.subplot(2, 1, 2)
                            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
                            S_dB = librosa.power_to_db(mel_spec, ref=np.max)
                            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax2, cmap='viridis')
                            ax2.set_title("Mel Spectrogram", fontsize=14, fontweight='bold', color='#1F2937')
                            ax2.set_facecolor('#F9FAFB')
                            cbar = fig.colorbar(img, ax=ax2, format='%+2.0f dB')
                            cbar.ax.tick_params(labelsize=10)
                            cbar.set_label('dB', rotation=270, labelpad=15, fontsize=12, color='#4B5563')
                            
                            # Add a modern touch to the figure
                            for spine in ax1.spines.values():
                                spine.set_color('#E5E7EB')
                            for spine in ax2.spines.values():
                                spine.set_color('#E5E7EB')
                                
                            # Set tick parameters
                            ax1.tick_params(colors='#6B7280', labelsize=10)
                            ax2.tick_params(colors='#6B7280', labelsize=10)
                            
                            plt.tight_layout()
                            
                            # Display the figure
                            st.pyplot(fig)
                            
                            # Add explanation of the visualizations
                            with st.expander("Understand the Audio Visualizations"):
                                st.markdown("""
                                ### Audio Visualization Explained
                                
                                #### 1. Audio Waveform (top)
                                The waveform shows how the amplitude of your audio changes over time. Peaks represent louder sounds.
                                
                                #### 2. Mel Spectrogram (bottom)
                                The spectrogram shows the frequency content over time, with colors representing intensity. 
                                This is what the AI model analyzes to detect emotions!
                                
                                - **Horizontal axis**: Time
                                - **Vertical axis**: Frequency (higher = higher pitch)
                                - **Color**: Intensity (brighter = louder)
                                """)
                        else:
                            st.warning("Audio data not available for visualization.")
                    except Exception as e:
                        st.error(f"Error displaying audio visualization: {e}")
    
    def display_about_section(self):
        """Display information about the application"""
        with st.container():
            colored_header(
                label="About this Application",
                description="How speech emotion detection works",
                color_name="violet-70"
            )
            
            # Modern hero section
            st.markdown("""
            <div style="background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
                        border-radius: 24px;
                        padding: 32px 24px;
                        margin-bottom: 32px;
                        text-align: center;">
                <div style="max-width: 800px; margin: 0 auto;">
                    <h2 style="font-weight: 700; color: #4338CA; margin-bottom: 16px;">
                        AI-Powered Speech Emotion Recognition
                    </h2>
                    <p style="font-size: 1.1rem; color: #4B5563; margin-bottom: 24px; line-height: 1.6;">
                        Our advanced deep learning system can identify human emotions from speech patterns,
                        helping you understand emotional context in audio recordings.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Create a modern process flow
            st.markdown("""
            <h3 style="font-weight: 600; color: #4F46E5; margin-bottom: 24px; font-size: 1.5rem;">How It Works</h3>
            <div style="display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 32px;">
                <div style="flex: 1; min-width: 220px; background-color: white; border-radius: 16px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 20px 25px -5px rgba(0,0,0,0.05); border: 1px solid rgba(226, 232, 240, 0.8);">
                    <div style="width: 48px; height: 48px; background: linear-gradient(135deg, #C7D2FE 0%, #A5B4FC 100%); border-radius: 12px; display: flex; align-items: center; justify-content: center; margin-bottom: 16px;">
                        <span style="font-size: 24px;">🔊</span>
                    </div>
                    <h4 style="font-weight: 600; color: #111827; margin-bottom: 12px;">1. Audio Processing</h4>
                    <p style="color: #6B7280; font-size: 0.95rem; line-height: 1.5;">Your audio is converted into a spectrogram - a visual representation of frequencies over time that captures vocal patterns.</p>
                </div>
                
                <div style="flex: 1; min-width: 220px; background-color: white; border-radius: 16px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 20px 25px -5px rgba(0,0,0,0.05); border: 1px solid rgba(226, 232, 240, 0.8);">
                    <div style="width: 48px; height: 48px; background: linear-gradient(135deg, #DBEAFE 0%, #BFDBFE 100%); border-radius: 12px; display: flex; align-items: center; justify-content: center; margin-bottom: 16px;">
                        <span style="font-size: 24px;">📊</span>
                    </div>
                    <h4 style="font-weight: 600; color: #111827; margin-bottom: 12px;">2. Feature Extraction</h4>
                    <p style="color: #6B7280; font-size: 0.95rem; line-height: 1.5;">The system extracts meaningful acoustic features like pitch, energy, tempo, and spectral patterns from your speech.</p>
                </div>
                
                <div style="flex: 1; min-width: 220px; background-color: white; border-radius: 16px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 20px 25px -5px rgba(0,0,0,0.05); border: 1px solid rgba(226, 232, 240, 0.8);">
                    <div style="width: 48px; height: 48px; background: linear-gradient(135deg, #E9D5FF 0%, #D8B4FE 100%); border-radius: 12px; display: flex; align-items: center; justify-content: center; margin-bottom: 16px;">
                        <span style="font-size: 24px;">🧠</span>
                    </div>
                    <h4 style="font-weight: 600; color: #111827; margin-bottom: 12px;">3. Neural Network Analysis</h4>
                    <p style="color: #6B7280; font-size: 0.95rem; line-height: 1.5;">A convolutional neural network (CNN) processes these features to identify emotional patterns similar to how humans recognize emotions.</p>
                </div>
                
                <div style="flex: 1; min-width: 220px; background-color: white; border-radius: 16px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 20px 25px -5px rgba(0,0,0,0.05); border: 1px solid rgba(226, 232, 240, 0.8);">
                    <div style="width: 48px; height: 48px; background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%); border-radius: 12px; display: flex; align-items: center; justify-content: center; margin-bottom: 16px;">
                        <span style="font-size: 24px;">📈</span>
                    </div>
                    <h4 style="font-weight: 600; color: #111827; margin-bottom: 12px;">4. Emotion Classification</h4>
                    <p style="color: #6B7280; font-size: 0.95rem; line-height: 1.5;">The system provides confidence scores for each possible emotion, helping you understand the likelihood of each emotional state.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Two column layout for emotions and accuracy
            col1, col2 = st.columns([1, 1])
            
            with col1:
                with stylable_container(
                    key="about_container1",
                    css_styles="""
                        {
                            background-color: white;
                            border-radius: 16px;
                            padding: 24px;
                            height: 100%;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 20px 25px -5px rgba(0,0,0,0.05);
                            border: 1px solid rgba(226, 232, 240, 0.8);
                        }
                    """
                ):
                    st.markdown("""
                    <h3 style="font-weight: 600; color: #4F46E5; margin-bottom: 16px; font-size: 1.3rem;">Emotions Detected</h3>
                    """, unsafe_allow_html=True)
                    
                    # Create a grid of emotion tags
                    emotions = [
                        {"name": "neutral", "icon": "😐", "description": "Balanced emotional state"},
                        {"name": "calm", "icon": "😌", "description": "Peaceful, relaxed state"},
                        {"name": "happy", "icon": "😄", "description": "Joyful, pleased"},
                        {"name": "sad", "icon": "😢", "description": "Unhappy, sorrowful"},
                        {"name": "angry", "icon": "😡", "description": "Annoyed, hostile"},
                        {"name": "fearful", "icon": "😨", "description": "Afraid, anxious"},
                        {"name": "disgust", "icon": "🤢", "description": "Revulsion, disapproval"}
                    ]
                    
                    for emotion in emotions:
                        color = EMOTION_COLORS.get(emotion["name"], "#607D8B")
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; margin-bottom: 12px; background-color: {color}22; border-radius: 8px; padding: 8px 12px;">
                            <div style="font-size: 1.25rem; margin-right: 12px;">{emotion["icon"]}</div>
                            <div>
                                <div style="font-weight: 600; color: {color};">{emotion["name"].capitalize()}</div>
                                <div style="font-size: 0.85rem; color: #6B7280;">{emotion["description"]}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                with stylable_container(
                    key="about_container2",
                    css_styles="""
                        {
                            background-color: white;
                            border-radius: 16px;
                            padding: 24px;
                            height: 100%;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 20px 25px -5px rgba(0,0,0,0.05);
                            border: 1px solid rgba(226, 232, 240, 0.8);
                        }
                    """
                ):
                    st.markdown("""
                    <h3 style="font-weight: 600; color: #4F46E5; margin-bottom: 16px; font-size: 1.3rem;">Accuracy Considerations</h3>
                    
                    <div style="margin-bottom: 16px;">
                        <h4 style="font-weight: 600; font-size: 1rem; margin-bottom: 8px;">Audio Quality</h4>
                        <div style="display: flex; align-items: center; margin-bottom: 4px;">
                            <div style="width: 100%; background-color: #E5E7EB; height: 8px; border-radius: 4px; margin-right: 8px;">
                                <div style="width: 90%; background: linear-gradient(90deg, #4F46E5, #7C3AED); height: 8px; border-radius: 4px;"></div>
                            </div>
                            <span style="font-size: 0.85rem; color: #6B7280;">High impact</span>
                        </div>
                        <p style="font-size: 0.9rem; color: #6B7280; margin-top: 4px;">Clearer audio with minimal background noise produces significantly better results.</p>
                    </div>
                    
                    <div style="margin-bottom: 16px;">
                        <h4 style="font-weight: 600; font-size: 1rem; margin-bottom: 8px;">Cultural Context</h4>
                        <div style="display: flex; align-items: center; margin-bottom: 4px;">
                            <div style="width: 100%; background-color: #E5E7EB; height: 8px; border-radius: 4px; margin-right: 8px;">
                                <div style="width: 75%; background: linear-gradient(90deg, #4F46E5, #7C3AED); height: 8px; border-radius: 4px;"></div>
                            </div>
                            <span style="font-size: 0.85rem; color: #6B7280;">Medium impact</span>
                        </div>
                        <p style="font-size: 0.9rem; color: #6B7280; margin-top: 4px;">Emotional expression varies across cultures and may affect recognition accuracy.</p>
                    </div>
                    
                    <div style="margin-bottom: 16px;">
                        <h4 style="font-weight: 600; font-size: 1rem; margin-bottom: 8px;">Speaker Variability</h4>
                        <div style="display: flex; align-items: center; margin-bottom: 4px;">
                            <div style="width: 100%; background-color: #E5E7EB; height: 8px; border-radius: 4px; margin-right: 8px;">
                                <div style="width: 80%; background: linear-gradient(90deg, #4F46E5, #7C3AED); height: 8px; border-radius: 4px;"></div>
                            </div>
                            <span style="font-size: 0.85rem; color: #6B7280;">High impact</span>
                        </div>
                        <p style="font-size: 0.9rem; color: #6B7280; margin-top: 4px;">Each person expresses emotions differently, which can affect recognition patterns.</p>
                    </div>
                    
                    <div>
                        <h4 style="font-weight: 600; font-size: 1rem; margin-bottom: 8px;">Training Data</h4>
                        <div style="display: flex; align-items: center; margin-bottom: 4px;">
                            <div style="width: 100%; background-color: #E5E7EB; height: 8px; border-radius: 4px; margin-right: 8px;">
                                <div style="width: 70%; background: linear-gradient(90deg, #4F46E5, #7C3AED); height: 8px; border-radius: 4px;"></div>
                            </div>
                            <span style="font-size: 0.85rem; color: #6B7280;">Medium impact</span>
                        </div>
                        <p style="font-size: 0.9rem; color: #6B7280; margin-top: 4px;">The system was trained on acted emotions which may differ from natural expressions.</p>
                    </div>
                </stylable_container>
            """)
            # Add tech stack info
            st.markdown("""
            <div style="margin-top: 32px; background-color: #F8FAFC; border-radius: 16px; padding: 24px; border: 1px solid #E2E8F0;">
                <h3 style="font-weight: 600; color: #0F172A; margin-bottom: 16px; font-size: 1.3rem;">Technical Information</h3>
                <div style="display: flex; flex-wrap: wrap; gap: 12px;">
                    <div style="background-color: #EFF6FF; color: #1E40AF; padding: 6px 12px; border-radius: 100px; font-size: 0.9rem; font-weight: 500;">Python 3.8+</div>
                    <div style="background-color: #EFF6FF; color: #1E40AF; padding: 6px 12px; border-radius: 100px; font-size: 0.9rem; font-weight: 500;">TensorFlow 2.x</div>
                    <div style="background-color: #EFF6FF; color: #1E40AF; padding: 6px 12px; border-radius: 100px; font-size: 0.9rem; font-weight: 500;">Librosa</div>
                    <div style="background-color: #EFF6FF; color: #1E40AF; padding: 6px 12px; border-radius: 100px; font-size: 0.9rem; font-weight: 500;">Streamlit</div>
                    <div style="background-color: #EFF6FF; color: #1E40AF; padding: 6px 12px; border-radius: 100px; font-size: 0.9rem; font-weight: 500;">CNN Architecture</div>
                    <div style="background-color: #EFF6FF; color: #1E40AF; padding: 6px 12px; border-radius: 100px; font-size: 0.9rem; font-weight: 500;">RAVDESS Dataset</div>
                </div>
                <p style="margin-top: 16px; color: #64748B; font-size: 0.95rem;">
                    This project uses a convolutional neural network trained on the RAVDESS dataset, with mel-spectrogram features extracted using Librosa.
                    The UI is built with Streamlit and the visualization pipeline uses Plotly and Matplotlib.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    def display_tensorboard_launcher(self):
        """Display TensorBoard launcher interface"""
        with st.container():
            colored_header(
                label="TensorBoard Visualization",
                description="Launch TensorBoard to visualize model training metrics",
                color_name="violet-70"
            )
            
            with stylable_container(
                key="tensorboard_container",
                css_styles="""
                    {
                        background-color: #f0f2f6;
                        border-radius: 10px;
                        padding: 20px;
                    }
                """
            ):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write("### TensorBoard Launcher")
                    st.write("""
                    TensorBoard provides interactive visualizations of model training metrics, 
                    architecture, and performance. Launch it to explore detailed insights into 
                    your emotion recognition model.
                    """)
                    
                    # Get log directory
                    logs_dir = st.text_input(
                        "Log Directory", 
                        value="logs",
                        help="Directory containing TensorFlow training logs"
                    )
                
                with col2:
                    st.write("### Launch")
                    tb_port = find_available_port()
                    if st.button("Start TensorBoard", key="start_tb"):
                        if os.path.exists(logs_dir):
                            process, port = start_tensorboard(logs_dir, tb_port)
                            if process:
                                self.tensorboard_process = process
                                self.tensorboard_port = port
                                st.session_state['tensorboard_running'] = True
                                st.success(f"TensorBoard started successfully on port {port}")
                        else:
                            st.error(f"Log directory {logs_dir} does not exist.")
            
            # Display TensorBoard iframe if running
            if st.session_state.get('tensorboard_running', False):
                with stylable_container(
                    key="tb_launcher",
                    css_styles="""
                        {
                            background-color: #f0e6ff;
                            border-radius: 8px;
                            padding: 15px;
                            border: 1px solid #d0c0ff;
                            margin-top: 10px;
                        }
                    """
                ):
                    st.markdown(f"""
                    ### TensorBoard is Running 🚀
                    
                    Open TensorBoard in a new browser tab: [http://localhost:{self.tensorboard_port}](http://localhost:{self.tensorboard_port})
                    """)
                    
                    if st.button("Stop TensorBoard", key="stop_tb"):
                        if self.tensorboard_process:
                            self.tensorboard_process.terminate()
                            self.tensorboard_process = None
                            st.session_state['tensorboard_running'] = False
                            st.success("TensorBoard stopped successfully")
    
    def run(self):
        """Main method to run the Streamlit application"""
        # Display modern app header with gradient text and design
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem; padding: 1.5rem 0;">
            <h1 class='main-header'>Speech Emotion Analyzer</h1>
            <p style="text-align: center; font-size: 1.1rem; color: #6B7280; max-width: 600px; margin: 0 auto 1.5rem auto; line-height: 1.5;">
                Detect emotions in speech using AI-powered deep learning technology
            </p>
            <div style="display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; margin-top: 8px;">
                <span style="background-color: #F3E8FF; color: #6B21A8; padding: 4px 12px; border-radius: 100px; font-size: 0.85rem; font-weight: 500;">Deep Learning</span>
                <span style="background-color: #E0F2FE; color: #0369A1; padding: 4px 12px; border-radius: 100px; font-size: 0.85rem; font-weight: 500;">CNN Architecture</span>
                <span style="background-color: #DCFCE7; color: #15803D; padding: 4px 12px; border-radius: 100px; font-size: 0.85rem; font-weight: 500;">Real-time Analysis</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # We could add a logo if it exists
        # if os.path.exists("assets/logo.png"):
        #     add_logo("assets/logo.png")
        
        # Sidebar navigation
        with st.sidebar:
            try:
                # Check if visualization image exists before trying to display it
                if os.path.exists("results/visualizations/enhanced_confusion_matrix.png"):
                    st.image("results/visualizations/enhanced_confusion_matrix.png", caption="Emotion Classification Matrix", use_container_width=True)
                else:
                    st.info("Visualization image not found. This won't affect the app's functionality.")
            except Exception as e:
                st.info(f"Could not load visualization image. The app will still function normally.")
            
            # Custom CSS for sidebar
            st.markdown("""
            <style>
                [data-testid="stSidebar"] {
                    background-color: #FFFFFF;
                    border-right: 1px solid #E2E8F0;
                    padding: 1rem;
                }
                
                .sidebar-header {
                    margin-bottom: 1rem;
                    text-align: center;
                }
                
                .sidebar-header img {
                    max-width: 100%;
                    border-radius: 8px;
                    margin-bottom: 0.5rem;
                }
                
                .info-box {
                    background-color: #F5F7FF;
                    border-radius: 12px;
                    padding: 16px;
                    margin-bottom: 20px;
                    border: 1px solid #E2E8F0;
                }
                
                .accuracy-meter {
                    height: 8px;
                    background-color: #E2E8F0;
                    border-radius: 4px;
                    margin-top: 8px;
                    margin-bottom: 12px;
                    overflow: hidden;
                }
                
                .accuracy-value {
                    height: 100%;
                    background: linear-gradient(90deg, #4F46E5, #7C3AED);
                    border-radius: 4px;
                    width: 80%;
                }
            </style>
            """, unsafe_allow_html=True)
            
            # Sidebar header with visualization
            st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
            # Display logo or visualization if available
            try:
                if os.path.exists("results/visualizations/enhanced_confusion_matrix.png"):
                    st.image("results/visualizations/enhanced_confusion_matrix.png", 
                            caption="Emotion Classification Matrix", 
                            use_container_width=True)
                else:
                    # Placeholder text if image not found
                    st.markdown('<div style="height: 100px; display: flex; align-items: center; justify-content: center; background-color: #F8FAFC; border-radius: 8px; margin-bottom: 12px;"><p style="color: #64748B; font-size: 0.9rem;">Visualization not available</p></div>', unsafe_allow_html=True)
            except Exception:
                # Handle any exception gracefully
                pass
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Modern navigation menu
            selected = option_menu(
                menu_title="Navigation",
                options=["Analyze Audio", "Visualization Dashboard", "View Examples", "TensorBoard", "About", "Settings"],
                icons=["mic-fill", "bar-chart-fill", "collection-play", "graph-up", "info-circle", "sliders"],
                menu_icon="menu-app",
                default_index=0,
                styles={
                    "container": {"padding": "0!important", "background-color": "transparent"},
                    "icon": {"color": "#4F46E5", "font-size": "18px"}, 
                    "nav-link": {
                        "font-size": "15px", 
                        "text-align": "left", 
                        "margin": "4px 0px", 
                        "padding": "10px 12px", 
                        "border-radius": "8px",
                        "--hover-color": "#F5F7FF",
                        "font-weight": "500"
                    },
                    "nav-link-selected": {
                        "background-color": "#4F46E5", 
                        "color": "white",
                        "font-weight": "600"
                    },
                }
            )
            
            # Model information with modern design
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown('<h4 style="font-size: 1rem; font-weight: 600; color: #1E293B; margin-bottom: 12px;">Model Information</h4>', unsafe_allow_html=True)
            
            st.markdown("""
            <div style="margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 4px;">
                    <span style="color: #64748B;">Model Type:</span>
                    <span style="color: #1E293B; font-weight: 500;">CNN</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 4px;">
                    <span style="color: #64748B;">Dataset:</span>
                    <span style="color: #1E293B; font-weight: 500;">RAVDESS</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.9rem;">
                    <span style="color: #64748B;">Accuracy:</span>
                    <span style="color: #1E293B; font-weight: 500;">~80%</span>
                </div>
                <div class="accuracy-meter">
                    <div class="accuracy-value"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Modern real-time processing toggle
            st.markdown("""
            <div style="background-color: #F5F7FF; border-radius: 12px; padding: 16px; margin-bottom: 20px; border: 1px solid #E2E8F0;">
                <h4 style="font-size: 1rem; font-weight: 600; color: #1E293B; margin-bottom: 12px;">Processing Settings</h4>
            """, unsafe_allow_html=True)
            
            if 'real_time_enabled' not in st.session_state:
                st.session_state.real_time_enabled = False
                
            real_time = st.toggle(
                "Real-time Analysis", 
                value=st.session_state.real_time_enabled,
                help="Process audio continuously for immediate feedback"
            )
            
            st.markdown("""
            <div style="font-size: 0.9rem; color: #64748B; margin-top: 8px;">
                <p>Real-time processing analyzes your voice as you speak</p>
            </div>
            </div>
            """, unsafe_allow_html=True)
            
            if real_time != st.session_state.real_time_enabled:
                st.session_state.real_time_enabled = real_time
                if real_time and not self.loaded:
                    self.load_model()
                self.real_time_processing = real_time
        
        # Display selected section
        if selected == "Analyze Audio":
            self.display_file_upload()
            
        elif selected == "Visualization Dashboard":
            colored_header(
                label="Emotion Analytics Dashboard",
                description="Visualize and analyze your emotion detection results",
                color_name="violet-70"
            )
            self.dashboard.display_dashboard()
            
        elif selected == "View Examples":
            self.display_demo_section()
            
        elif selected == "TensorBoard":
            self.display_tensorboard_launcher()
            
        elif selected == "About":
            self.display_about_section()
            
        elif selected == "Settings":
            colored_header(
                label="Settings",
                description="Configure application settings",
                color_name="violet-70"
            )
            
            # Create tabs for different settings categories
            settings_tabs = st.tabs(["🧠 Model", "🔊 Audio", "🖥️ Interface", "ℹ️ About"])
            
            with settings_tabs[0]:
                with stylable_container(
                    key="model_settings_container",
                    css_styles="""
                        {
                            background-color: #FFFFFF;
                            border-radius: 16px;
                            padding: 24px;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 20px 25px -5px rgba(0,0,0,0.05);
                            margin-top: 16px;
                            margin-bottom: 16px;
                            border: 1px solid rgba(226, 232, 240, 0.8);
                        }
                    """
                ):
                    st.markdown("<h3 style='font-weight: 600; color: #4F46E5; margin-bottom: 16px;'>Model Settings</h3>", unsafe_allow_html=True)
                    
                    # Create two columns for better layout
                    model_col1, model_col2 = st.columns([3, 2])
                    
                    with model_col1:
                        model_path = st.text_input("Model Path", 
                                                value=self.model_path,
                                                help="Path to the trained emotion classification model")
                        
                        model_type = st.selectbox(
                            "Model Type", 
                            options=["CNN", "MLP"],
                            index=0,
                            help="Select the type of neural network architecture"
                        )
                    
                    with model_col2:
                        st.markdown("""
                        <div style="background-color: #F0F7FF; border-radius: 12px; padding: 16px; margin-top: 24px;">
                            <h4 style="font-weight: 600; color: #1E40AF; font-size: 0.9rem; margin-bottom: 12px;">Model Information</h4>
                            <ul style="margin-left: 0; padding-left: 20px; font-size: 0.9rem; color: #334155;">
                                <li>CNN models work better with spectrograms</li>
                                <li>Model files should be in .keras or .h5 format</li>
                                <li>Default model is pre-trained on RAVDESS dataset</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
            
            with settings_tabs[1]:
                with stylable_container(
                    key="audio_settings_container",
                    css_styles="""
                        {
                            background-color: #FFFFFF;
                            border-radius: 16px;
                            padding: 24px;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 20px 25px -5px rgba(0,0,0,0.05);
                            margin-top: 16px;
                            margin-bottom: 16px;
                            border: 1px solid rgba(226, 232, 240, 0.8);
                        }
                    """
                ):
                    st.markdown("<h3 style='font-weight: 600; color: #4F46E5; margin-bottom: 16px;'>Audio Processing</h3>", unsafe_allow_html=True)
                    
                    audio_col1, audio_col2 = st.columns([3, 2])
                    
                    with audio_col1:
                        sample_length = st.slider(
                            "Sample Length (seconds)", 
                            min_value=1, 
                            max_value=10, 
                            value=5,
                            help="Length of audio to analyze (longer samples will be trimmed)"
                        )
                        
                        sample_rate = st.select_slider(
                            "Sample Rate (Hz)",
                            options=[8000, 16000, 22050, 44100],
                            value=16000,
                            help="Audio sampling rate for processing"
                        )
                        
                        feature_type = st.selectbox(
                            "Feature Type", 
                            options=["Mel Spectrogram", "MFCC", "Raw Waveform"],
                            index=0,
                            help="Type of features to extract from audio"
                        )
                    
                    with audio_col2:
                        st.markdown("""
                        <div style="background-color: #F0FDF4; border-radius: 12px; padding: 16px; margin-top: 24px;">
                            <h4 style="font-weight: 600; color: #15803D; font-size: 0.9rem; margin-bottom: 12px;">Audio Tips</h4>
                            <ul style="margin-left: 0; padding-left: 20px; font-size: 0.9rem; color: #334155;">
                                <li>5-second clips work best for emotion detection</li>
                                <li>16kHz is the recommended sample rate</li>
                                <li>Mel Spectrograms generally provide the best results</li>
                                <li>Clear speech with minimal background noise yields better accuracy</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
            
            with settings_tabs[2]:
                with stylable_container(
                    key="interface_settings_container",
                    css_styles="""
                        {
                            background-color: #FFFFFF;
                            border-radius: 16px;
                            padding: 24px;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 20px 25px -5px rgba(0,0,0,0.05);
                            margin-top: 16px;
                            margin-bottom: 16px;
                            border: 1px solid rgba(226, 232, 240, 0.8);
                        }
                    """
                ):
                    st.markdown("<h3 style='font-weight: 600; color: #4F46E5; margin-bottom: 16px;'>Interface Settings</h3>", unsafe_allow_html=True)
                    
                    ui_col1, ui_col2 = st.columns([1, 1])
                    
                    with ui_col1:
                        theme = st.selectbox(
                            "Color Theme", 
                            options=["Default", "Dark", "Light", "High Contrast"],
                            index=0,
                            help="Select the application color theme"
                        )
                        
                        charts_type = st.radio(
                            "Chart Type",
                            options=["Interactive", "Static"],
                            index=0,
                            horizontal=True,
                            help="Choose between interactive or static visualizations"
                        )
                        
                        show_advanced = st.toggle("Show Advanced Options", value=False)
                    
                    with ui_col2:
                        auto_record = st.toggle("Auto-Start Recording", value=False,
                                      help="Automatically start recording when switching to the Record tab")
                        
                        show_debug = st.toggle("Debug Information", value=False,
                                    help="Show detailed technical information for debugging purposes")
                        
                        if show_advanced:
                            st.selectbox(
                                "Plot Style", 
                                options=["Default", "Modern", "Classic", "Minimal"],
                                index=1,
                                help="Visual style for plots and charts"
                            )
            
            with settings_tabs[3]:
                with stylable_container(
                    key="about_settings_container",
                    css_styles="""
                        {
                            background-color: #FFFFFF;
                            border-radius: 16px;
                            padding: 24px;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 20px 25px -5px rgba(0,0,0,0.05);
                            margin-top: 16px;
                            margin-bottom: 16px;
                            border: 1px solid rgba(226, 232, 240, 0.8);
                        }
                    """
                ):
                    st.markdown("<h3 style='font-weight: 600; color: #4F46E5; margin-bottom: 16px;'>About This Application</h3>", unsafe_allow_html=True)
                    
                    st.markdown("""
                    #### Speech Emotion Analyzer v1.1
                    
                    This application uses deep learning to analyze emotional content in speech recordings.
                    
                    **Built with**:
                    - Python 3.8+
                    - TensorFlow 2.x
                    - Streamlit 1.x
                    - Librosa 0.9+
                    
                    **Developer**: AI Research Team
                    
                    **Last Updated**: May 2025
                    """)
                    
                    st.markdown("---")
                    
                    st.markdown("""
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <a href="#" style="text-decoration: none; color: #4F46E5;">Documentation</a>
                        </div>
                        <div>
                            <a href="#" style="text-decoration: none; color: #4F46E5;">Source Code</a>
                        </div>
                        <div>
                            <a href="#" style="text-decoration: none; color: #4F46E5;">Report Issues</a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Save button below tabs
            with stylable_container(
                key="settings_save_container",
                css_styles="""
                    {
                        background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
                        border-radius: 16px;
                        padding: 20px;
                        margin-top: 16px;
                        text-align: center;
                        border: 1px solid rgba(226, 232, 240, 0.8);
                    }
                """
            ):
                save_col1, save_col2, save_col3 = st.columns([1, 2, 1])
                with save_col2:
                    if st.button("Save Settings", use_container_width=True):
                        self.model_path = model_path
                        # Add other settings here as needed
                        st.success("✅ Settings saved successfully!")
                        st.info("Some changes may require restarting the application to take effect.")
                        self.loaded = False  # Force model reload with new settings
        
        # Modern footer with better styling and information
        st.markdown("""
        <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #E5E7EB;">
            <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;">
                <div style="margin-bottom: 16px;">
                    <p style="color: #6B7280; font-size: 0.9rem; font-weight: 500; margin-bottom: 4px;">
                        Speech Emotion Analyzer v1.1
                    </p>
                    <p style="color: #9CA3AF; font-size: 0.8rem; margin: 0;">
                        © 2025 AI Research Team
                    </p>
                </div>
                <div style="display: flex; gap: 24px; margin-bottom: 16px;">
                    <a href="#" style="color: #6B7280; text-decoration: none; font-size: 0.9rem; transition: color 0.2s;">Help</a>
                    <a href="#" style="color: #6B7280; text-decoration: none; font-size: 0.9rem; transition: color 0.2s;">Privacy</a>
                    <a href="#" style="color: #6B7280; text-decoration: none; font-size: 0.9rem; transition: color 0.2s;">Terms</a>
                    <a href="#" style="color: #6B7280; text-decoration: none; font-size: 0.9rem; transition: color 0.2s;">Contact</a>
                </div>
            </div>
            <div style="text-align: center; margin-top: 16px;">
                <p style="color: #9CA3AF; font-size: 0.8rem;">
                    Made with <span style="color: #EF4444;">❤️</span> using Streamlit and TensorFlow
                </p>
            </div>
        </footer>
        """, unsafe_allow_html=True)

# Initialize session state
if 'tensorboard_running' not in st.session_state:
    st.session_state['tensorboard_running'] = False

if 'real_time_enabled' not in st.session_state:
    st.session_state['real_time_enabled'] = False

if __name__ == "__main__":
    app = EmotionAnalyzer()
    app.run()