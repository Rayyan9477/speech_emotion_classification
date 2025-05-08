import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Set page configuration
st.set_page_config(
    page_title="Speech Emotion Analyzer",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4527A0;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #5E35B1;
        margin-bottom: 1rem;
    }
    .card-container {
        border-radius: 10px;
        padding: 20px;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .result-container {
        background-color: #EDE7F6;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    .emotion-tag {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 15px;
        margin-right: 5px;
        font-weight: bold;
        color: white;
    }
    .stProgress .st-bo {
        background-color: #5E35B1;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        background-color: #EDE7F6;
        border-radius: 5px 5px 0 0;
        gap: 1rem;
        padding: 1rem;
    }
    .gauge-chart {
        margin: 0 auto;
        text-align: center;
    }
    .stAlert {
        transition: all 0.3s ease-in-out;
    }
    .css-nahz7x {
        transform: scale(1.02);
        transition: transform 0.3s ease;
    }
    .css-nahz7x:hover {
        transform: scale(1.05);
    }
    .tb-launcher {
        background-color: #f0e6ff;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #d0c0ff;
        margin-top: 10px;
    }
    .real-time-indicator {
        color: #4CAF50;
        font-weight: bold;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% {
            opacity: 0.6;
        }
        50% {
            opacity: 1;
        }
        100% {
            opacity: 0.6;
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
        self.model_path = "models/cnn_emotion_model.keras"
        self.backup_model_path = "models/cnn_emotion_model.h5"
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        self.loaded = False
        self.upload_folder = "uploads"
        self.tensorboard_process = None
        self.tensorboard_port = None
        self.real_time_processing = False
        self.processing_thread = None
        self.tensorflow_available = tensorflow_available
        self.ensure_upload_dir()
        
        # Check if demo folder exists, create if not
        os.makedirs("demo_files", exist_ok=True)
        
        # Check if model folder exists, create if not
        os.makedirs("models", exist_ok=True)
        
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
        """Create a gauge chart to show confidence level"""
        try:
            color = EMOTION_COLORS.get(emotion, "#607D8B")
            
            # Create color variants for steps without trying to parse hex colors
            light_color = color  # Base color
            medium_color = "#BDBDBD"  # Medium gray for middle range
            light_gray = "#E0E0E0"  # Light gray for low range
            light_variant = "rgba(200, 200, 200, 0.3)"  # Safe light variant
            
            # Try to create hex-based color variant, but use safe fallback if it fails
            try:
                # Parse hex color into RGB components
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
                light_variant = f"rgba({r}, {g}, {b}, 0.3)"
            except:
                # If hex parsing fails, use the safe fallback
                pass
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": f"{emotion.capitalize()} Confidence", "font": {"size": 24, "color": color}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                    "bar": {"color": color},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "gray",
                    "steps": [
                        {"range": [0, 30], "color": light_gray},
                        {"range": [30, 70], "color": medium_color},
                        {"range": [70, 100], "color": light_variant}
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 90
                    }
                }
            ))
            
            fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "darkblue", "family": "Arial"}
            )
            
            return fig
        except Exception as e:
            print(f"Error creating gauge chart: {e}")
            import traceback
            print(traceback.format_exc())
            
            # Return a simple fallback figure in case of error
            fallback_fig = go.Figure()
            fallback_fig.add_annotation(text="Chart creation failed", showarrow=False)
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
                            background-color: #f0f2f6;
                            border-radius: 10px;
                            padding: 20px;
                            margin-top: 10px;
                        }
                    """
                ):
                    st.markdown("### Upload Audio File")
                    st.markdown("Upload a WAV or MP3 file of someone speaking to analyze their emotional state.")
                    
                    uploaded_file = st.file_uploader(
                        "Choose an audio file (WAV or MP3)", 
                        type=["wav", "mp3"],
                        help="Upload a short audio clip (ideally 5-10 seconds) of someone speaking"
                    )
                    
                    st.markdown(
                        """
                        ### Tips for good results:
                        - Use clear audio with minimal background noise
                        - Short clips (5-10 seconds) work best
                        - Make sure the speaker's voice is prominent
                        """
                    )
                    
                    # Display example upload button
                    if st.button("Don't have a file? Try a sample", key="try_sample"):
                        if os.path.exists("demo_files/happy_sample.wav"):
                            self.process_audio("demo_files/happy_sample.wav")
                            st.success("Loaded sample audio!")
                        else:
                            st.warning("Demo files not found. Please go to the 'View Examples' section.")
            
            with tab2:
                with stylable_container(
                    key="record_container",
                    css_styles="""
                        {
                            background-color: #ede7f6;
                            border-radius: 10px;
                            padding: 20px;
                            margin-top: 10px;
                        }
                    """
                ):
                    st.markdown("### Record Audio")
                    st.markdown("Record your voice directly in the browser to analyze emotions in real-time.")
                    
                    # Add a toggle for real-time processing
                    real_time_enabled = st.toggle("Enable real-time analysis", value=False, 
                                           help="Process audio as you speak for immediate feedback")
                    
                    if real_time_enabled:
                        st.markdown("<p class='real-time-indicator'>● Real-time analysis active</p>", unsafe_allow_html=True)
                        if not self.loaded:
                            self.load_model()
                        self.real_time_processing = True
                    else:
                        self.real_time_processing = False
                    
                    # Check if audio recorder component is available
                    if not audiorecorder_available:
                        st.error("Audio recorder component is not available. Please use the Upload option instead.")
                        st.info("If you want to use the recording feature, run: 'pip install streamlit-audiorecorder'")
                    else:
                        st.markdown("Click below to start/stop recording:")
                        # Use the st_audiorecorder component
                        audio_recording = st_audiorecorder(
                            text="Click to record",
                            recording_color="#e65100",
                            neutral_color="#5e35b1",
                            sample_rate=16000,
                        )
                    
                    # If real-time enabled, create a placeholder for live results
                    if real_time_enabled and audiorecorder_available:
                        live_result_placeholder = st.empty()
                        
                        # Check if there are any results in the queue
                        if not result_queue.empty():
                            emotion, confidence_scores, audio_y, audio_sr = result_queue.get()
                            with live_result_placeholder.container():
                                st.markdown(f"### Live Emotion: **{emotion.upper()}**")
                                max_confidence = confidence_scores.get(emotion, 0)
                                gauge_fig = self.create_gauge_chart(max_confidence, emotion)
                                st.plotly_chart(gauge_fig, use_container_width=True)
            
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
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                with stylable_container(
                    key="demo1",
                    css_styles="""
                        {
                            background-color: #e3f2fd;
                            border-radius: 10px;
                            padding: 15px;
                        }
                    """
                ):
                    st.write("### Happy Example")
                    if st.button("Analyze Happy Sample", key="happy_btn"):
                        # Use a sample file from the dataset
                        sample_path = "demo_files/happy_sample.wav"
                        if not os.path.exists(sample_path):
                            st.warning("Demo file not found. Please run the setup script first.")
                        else:
                            self.process_audio(sample_path)
            
            with col2:
                with stylable_container(
                    key="demo2",
                    css_styles="""
                        {
                            background-color: #e8eaf6;
                            border-radius: 10px;
                            padding: 15px;
                        }
                    """
                ):
                    st.write("### Angry Example")
                    if st.button("Analyze Angry Sample", key="angry_btn"):
                        # Use a sample file from the dataset
                        sample_path = "demo_files/angry_sample.wav"
                        if not os.path.exists(sample_path):
                            st.warning("Demo file not found. Please run the setup script first.")
                        else:
                            self.process_audio(sample_path)
            
            with col3:
                with stylable_container(
                    key="demo3",
                    css_styles="""
                        {
                            background-color: #f3e5f5;
                            border-radius: 10px;
                            padding: 15px;
                        }
                    """
                ):
                    st.write("### Sad Example")
                    if st.button("Analyze Sad Sample", key="sad_btn"):
                        # Use a sample file from the dataset
                        sample_path = "demo_files/sad_sample.wav"
                        if not os.path.exists(sample_path):
                            st.warning("Demo file not found. Please run the setup script first.")
                        else:
                            self.process_audio(sample_path)
    
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
            
            col1, col2 = st.columns([2, 3])
            
            with col1:
                with stylable_container(
                    key="result_container",
                    css_styles=f"""
                        {{
                            background-color: {EMOTION_COLORS.get(emotion, "#607D8B") + "22"};
                            border-radius: 10px;
                            padding: 20px;
                            border: 2px solid {EMOTION_COLORS.get(emotion, "#607D8B")};
                        }}
                    """
                ):
                    st.markdown(f"### Detected Emotion: <span style='color:{EMOTION_COLORS.get(emotion, '#607D8B')}'>{emotion.upper()}</span>", unsafe_allow_html=True)
                    
                    # Check if confidence_scores is empty, and handle it gracefully
                    if not confidence_scores:
                        st.warning("No confidence scores available. The model may not have produced valid predictions.")
                        max_confidence = 0
                    else:
                        # Display most confident emotion with gauge chart
                        max_confidence = max(confidence_scores.values())
                    
                    try:
                        gauge_fig = self.create_gauge_chart(max_confidence, emotion)
                        if gauge_fig is not None:
                            st.plotly_chart(gauge_fig, use_container_width=True)
                        else:
                            st.warning("Could not create confidence gauge chart.")
                    except Exception as e:
                        st.error(f"Error displaying gauge chart: {e}")
                    
                    # Display emotion description
                    st.markdown(f"**Description**: {EMOTION_DESCRIPTIONS.get(emotion, '')}")
                    
                    # Display original audio for playback
                    try:
                        st.audio(audio_file_path)
                    except Exception as e:
                        st.error(f"Error playing audio: {e}")
            
            with col2:
                tabs = st.tabs(["Confidence Scores", "Audio Visualization"])
                
                with tabs[0]:
                    # Check if confidence_scores is empty
                    if not confidence_scores:
                        st.warning("No confidence scores available to display.")
                    else:
                        try:
                            # Display interactive bar chart of confidence scores
                            fig = self.create_interactive_visualization(confidence_scores)
                            if fig is not None:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Could not create confidence visualization.")
                        except Exception as e:
                            st.error(f"Error displaying confidence scores visualization: {e}")
                
                with tabs[1]:
                    try:
                        # Display audio visualizations if y and sr are available
                        if y is not None and sr is not None:
                            fig = self.visualize_audio(y, sr)
                            if fig is not None:
                                st.pyplot(fig)
                            else:
                                st.warning("Could not visualize audio waveform.")
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
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                with stylable_container(
                    key="about_container1",
                    css_styles="""
                        {
                            background-color: #f5f5f5;
                            border-radius: 10px;
                            padding: 20px;
                        }
                    """
                ):
                    st.markdown("""
                    ### How it Works
                    
                    1. **Audio Processing**: Your audio is converted into a spectrogram - a visual representation of frequencies over time.
                    
                    2. **Feature Extraction**: The system extracts relevant features from the spectrogram that help identify emotional patterns.
                    
                    3. **Neural Network Analysis**: A convolutional neural network (CNN) analyzes these features and classifies the emotion.
                    
                    4. **Confidence Scoring**: The system assigns confidence scores indicating how certain it is about each possible emotion.
                    """)
            
            with col2:
                with stylable_container(
                    key="about_container2",
                    css_styles="""
                        {
                            background-color: #f5f5f5;
                            border-radius: 10px;
                            padding: 20px;
                        }
                    """
                ):
                    st.markdown("""
                    ### Emotions Detected
                    
                    This system can detect 7 different emotions:
                    
                    - **Neutral**: Balanced emotional state
                    - **Calm**: Peaceful, relaxed state
                    - **Happy**: Joyful, pleased
                    - **Sad**: Unhappy, sorrowful
                    - **Angry**: Annoyed, hostile
                    - **Fearful**: Afraid, anxious
                    - **Disgust**: Revulsion, strong disapproval
                    """)
            
            st.markdown("""
            ### Accuracy Considerations
            
            The accuracy of speech emotion recognition depends on several factors:
            
            - **Audio Quality**: Clearer audio produces better results
            - **Cultural Context**: Emotional expression varies across cultures
            - **Speaker Variability**: Each person expresses emotions differently
            - **Acted vs. Natural**: The system was trained on acted emotions which may differ from natural expressions
            """)
    
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
        # Display app header
        st.markdown("<h1 class='main-header'>Speech Emotion Analyzer</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 1.2rem;'>Detect emotions in speech using AI</p>", unsafe_allow_html=True)
        
        # Add logo
        # add_logo("assets/logo.png")
        
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
            
            selected = option_menu(
                menu_title="Navigation",
                options=["Analyze Audio", "View Examples", "TensorBoard", "About", "Settings"],
                icons=["soundwave", "play-circle", "graph-up", "info-circle", "gear"],
                menu_icon="cast",
                default_index=0,
                styles={
                    "container": {"padding": "5!important", "background-color": "#f0f2f6"},
                    "icon": {"color": "purple", "font-size": "25px"}, 
                    "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                    "nav-link-selected": {"background-color": "#5E35B1"},
                }
            )
            
            st.markdown("---")
            st.markdown("### Model Information")
            st.info("""
            - Model Type: CNN
            - Dataset: RAVDESS
            - Accuracy: ~75-85% on test data
            """)
            
            # Add real-time toggle in sidebar
            st.markdown("### Real-time Processing")
            if 'real_time_enabled' not in st.session_state:
                st.session_state.real_time_enabled = False
                
            real_time = st.checkbox(
                "Enable real-time analysis", 
                value=st.session_state.real_time_enabled,
                help="Process audio continuously for immediate feedback"
            )
            
            if real_time != st.session_state.real_time_enabled:
                st.session_state.real_time_enabled = real_time
                if real_time and not self.loaded:
                    self.load_model()
                self.real_time_processing = real_time
        
        # Display selected section
        if selected == "Analyze Audio":
            self.display_file_upload()
            
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
            
            with stylable_container(
                key="settings_container",
                css_styles="""
                    {
                        background-color: #f5f5f5;
                        border-radius: 10px;
                        padding: 20px;
                    }
                """
            ):
                st.write("### Model Settings")
                model_path = st.text_input("Model Path", value=self.model_path)
                
                st.write("### Audio Processing Settings")
                sample_length = st.slider("Sample Length (seconds)", 1, 10, 5)
                
                if st.button("Save Settings"):
                    self.model_path = model_path
                    st.success("Settings saved successfully!")
                    self.loaded = False  # Force model reload with new settings
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<p style='text-align: center; color: gray;'>Speech Emotion Analyzer v1.1 | Made with ❤️ using Streamlit</p>",
            unsafe_allow_html=True
        )

# Initialize session state
if 'tensorboard_running' not in st.session_state:
    st.session_state['tensorboard_running'] = False

if 'real_time_enabled' not in st.session_state:
    st.session_state['real_time_enabled'] = False

if __name__ == "__main__":
    app = EmotionAnalyzer()
    app.run()