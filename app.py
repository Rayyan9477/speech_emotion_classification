#!/usr/bin/env python3
"""Streamlit UI entrypoint (simplified – Streamlit only). Run with: streamlit run app.py"""

import sys
import os
import time
from pathlib import Path
import logging
import json

def is_streamlit_runtime() -> bool:
    """Return True if running under Streamlit; guards optional code."""
    try:  # lightweight detection
        import streamlit as _st  # noqa
        return True
    except Exception:
        return False


# =======================
# Streamlit UI (merged)
# =======================
import streamlit as st
st.set_page_config(
    page_title="Speech Emotion Analyzer",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the project root to Python path (now this file is already at project root)
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import monkey patch first to fix OverflowError
from src.utils.monkey_patch import monkeypatch
monkeypatch()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import TensorFlow with error handling (catch broad errors e.g., DLL issues)
try:
    import tensorflow as tf
    tensorflow_available = True
    tensorflow_error = ""
except Exception as e:
    tensorflow_available = False
    tensorflow_error = str(e)

import librosa
import librosa.display
import soundfile as sf  # noqa: F401 (kept for parity; may be used elsewhere)
import subprocess as _subprocess
import threading
import queue  # noqa: F401 (reserved for future RT audio)
from streamlit_option_menu import option_menu
from streamlit_extras.colored_header import colored_header  # noqa: F401
from streamlit_extras.app_logo import add_logo  # noqa: F401
from streamlit_extras.card import card  # noqa: F401
from streamlit_extras.stylable_container import stylable_container
import plotly.express as px  # noqa: F401
import tempfile
import webbrowser
import socket
import plotly.graph_objects as go
import io  # noqa: F401

# Import custom modules
from src.models.emotion_model import EmotionModel  # noqa: F401
from src.features.feature_extractor import FeatureExtractor
from src.ui.dashboard import EmotionDashboard
from src.models.model_manager import ModelManager as DriverModelManager
from src.main import train_model, TrainArgs
import portalocker
from src.core import config as core_config
try:
    from audio_recorder_streamlit import audio_recorder
except Exception:
    audio_recorder = None

 

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
        del st.session_state.training_status
    elif status['type'] == 'error':
        status_container.markdown(f"""
        <div class='status-container status-error'>
            <h4>❌ Training failed!</h4>
            <p>{status['message']}</p>
        </div>
        """, unsafe_allow_html=True)
        del st.session_state.training_status


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
    """Main class for the Emotion Analysis Application"""

    def __init__(self):
        """Initialize the Emotion Analyzer application"""
    # Setup paths with proper permissions handling
        try:
            upload_dir = Path('uploads')
            upload_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using upload directory: {upload_dir.resolve()}")
        except PermissionError as pe:
            logger.error(f"Permission denied creating upload directory: {pe}")
            upload_dir = Path(tempfile.gettempdir()) / 'speech_emotion_uploads'
            # This mkdir may still fail under tests due to monkeypatch, but we still use the fallback path value
            try:
                upload_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            st.warning(f"Using fallback temp directory: {upload_dir}")
        # Use the resolved `upload_dir` for `upload_folder`
        self.upload_folder = str(upload_dir)
        self.model_path = "models/emotion_model"
        self.backup_model_path = "models/emotion_model.h5"

        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.dashboard = EmotionDashboard()
        self.model_manager = DriverModelManager()

        # Set default emotion labels
        # Centralize labels from configuration
        try:
            self.emotion_labels = core_config.Config().training.emotion_labels
        except Exception:
            self.emotion_labels = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

        # Use central config model paths
        try:
            self.model_path = core_config.Config().models.cnn.model_path
            self.backup_model_path = core_config.Config().models.cnn.backup_path
        except Exception:
            pass

        # Internal state
        self.loaded = False
        self.model = None
        self.processing_thread = None
        self.real_time_processing = False
        self.last_prediction = None
        self.tensorboard_process = None
        self.tensorboard_port = 6006
        self.training_in_progress = False
        self.training_process = None

        # Check tensorflow availability
        self.tensorflow_available = tensorflow_available

        # Ensure upload directory exists
        self.ensure_upload_dir()

    def _inject_css(self):
        """Inject external CSS styles for a consistent UI/UX."""
        try:
            css_path = Path(__file__).parent / 'src' / 'ui' / 'assets' / 'styles.css'
            if css_path.exists():
                with open(css_path, 'r', encoding='utf-8') as f:
                    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        except Exception:
            # Non-fatal styling error
            pass

    def ensure_upload_dir(self):
        """Ensure the upload directory exists"""
        try:
            os.makedirs(self.upload_folder, mode=0o755, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create upload directory: {e}")
            st.error(f"Cannot create upload directory: {e}")
            raise

    def load_model(self):
        """Load an existing model; if none exists, start a single guarded training run.

        Returns True when a model is ready, otherwise False.
        """
        if self.model is not None:
            return True
        if not self.tensorflow_available:
            st.error(f"TensorFlow is not available. Error: {tensorflow_error}")
            st.info("Please reinstall TensorFlow or fix the DLL loading issue to use this application.")
            return False

        @st.cache_resource(show_spinner=False)
        def _cached_load(path: str):
            return tf.keras.models.load_model(path)

        try:
            # Step 1: Try to load from model registry (most recent)
            try:
                latest = self.model_manager.get_latest_model(model_type="cnn") or self.model_manager.get_latest_model(None)
                if latest and latest.get('path'):
                    model_path = latest['path']
                    if os.path.exists(model_path):
                        try:
                            self.model = _cached_load(model_path)
                            self.model_path = model_path
                            self.loaded = True
                            feature_info = self.model_manager.load_feature_info(model_path=model_path) or {}
                            norm_params = feature_info.get('normalization_params')
                            if norm_params:
                                self.feature_extractor.set_normalization_params(norm_params)
                            st.success("✅ Model loaded from registry!")
                            logger.info(f"Successfully loaded model from registry: {model_path}")
                            return True
                        except Exception as e:
                            logger.warning(f"Failed to load model from registry path {model_path}: {e}")
            except Exception as e:
                logger.warning(f"Error accessing model registry: {e}")

            # Step 2: Try known model paths (prioritizing corrected models)
            model_paths_to_try = [
                "models/cnn_emotion_model_20251022_065208.keras",  # Latest multimodal model
                self.model_path,
                "models/cnn_emotion_model.keras"
            ]

            for model_path in model_paths_to_try:
                if os.path.exists(model_path):
                    try:
                        self.model = _cached_load(model_path)
                        self.model_path = model_path
                        self.loaded = True
                        try:
                            feature_info = self.model_manager.load_feature_info(model_path=model_path) or {}
                            norm_params = feature_info.get('normalization_params')
                            if norm_params:
                                self.feature_extractor.set_normalization_params(norm_params)
                        except Exception as e:
                            logger.warning(f"Could not load feature info: {e}")
                        st.success("✅ Model loaded successfully!")
                        logger.info(f"Successfully loaded model from: {model_path}")
                        return True
                    except Exception as e:
                        logger.warning(f"Failed to load model from {model_path}: {e}")
                        continue

            # Step 3: Scan logs directory for recent models
            log_dir = Path('logs')
            if log_dir.exists():
                run_dirs = sorted(log_dir.glob('run_*'), reverse=True)
                for run_dir in run_dirs[:5]:  # Check last 5 runs
                    model_candidates = [
                        run_dir / 'best_model.keras',
                        run_dir / 'cnn_emotion_model.keras',
                        run_dir / 'emotion_model.keras'
                    ]
                    for candidate in model_candidates:
                        if candidate.exists():
                            try:
                                self.model = _cached_load(str(candidate))
                                self.model_path = str(candidate)
                                self.loaded = True
                                feature_info = self.model_manager.load_feature_info(model_path=str(candidate)) or {}
                                norm_params = feature_info.get('normalization_params')
                                if norm_params:
                                    self.feature_extractor.set_normalization_params(norm_params)
                                st.success("✅ Model loaded from training logs!")
                                logger.info(f"Successfully loaded model from logs: {candidate}")
                                return True
                            except Exception as e:
                                logger.warning(f"Failed to load model from {candidate}: {e}")
                                continue

            # Step 4: Scan models directory for any available models
            models_dir = Path('models')
            if models_dir.exists():
                model_files = list(models_dir.glob('*.keras')) + list(models_dir.glob('*.h5'))
                model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)  # Most recent first

                for model_file in model_files[:5]:  # Try last 5 models
                    try:
                        self.model = _cached_load(str(model_file))
                        self.model_path = str(model_file)
                        self.loaded = True
                        try:
                            feature_info = self.model_manager.load_feature_info(model_path=str(model_file)) or {}
                            norm_params = feature_info.get('normalization_params')
                            if norm_params:
                                self.feature_extractor.set_normalization_params(norm_params)
                        except Exception as e:
                            logger.warning(f"Could not load feature info for {model_file}: {e}")
                        st.success("✅ Model loaded from models directory!")
                        logger.info(f"Successfully loaded model from models directory: {model_file}")
                        return True
                    except Exception as e:
                        logger.warning(f"Failed to load model from {model_file}: {e}")
                        continue

            # Step 5: If no model found, start automatic training
            lock_file = Path('models') / 'training.lock'
            if lock_file.exists():
                self.training_in_progress = True
                st.session_state.training_status = {
                    'type': 'progress',
                    'message': 'Model training is already in progress. Your upload will be analyzed automatically when ready.'
                }
                logger.info("Training already in progress, waiting...")
                return False

            logger.info("No trained model found, starting automatic training...")
            st.info("📊 No trained model found. Starting automatic model training...")

            # Check if we have demo data to train on
            demo_files_exist = self._check_demo_files_available()
            if not demo_files_exist:
                st.warning("⚠️ No training data found. Please ensure demo files exist or provide training data.")
                st.info("💡 Tip: The system needs audio files in the 'demo_files' directory to train a model.")
                return False

            return self.train_model_automatically()

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            st.error(f"❌ Error loading model: {str(e)}")
            return False

    def extract_features(self, audio_file_path):
        """Extract features from audio file for model prediction using FeatureExtractor."""
        try:
            with st.spinner("Extracting audio features..."):
                # Load audio file
                y, sr = librosa.load(audio_file_path, sr=None)
                duration = len(y) / float(sr)

                # Intelligent audio processing for long files
                try:
                    from src.core.config import Config
                    config_duration = Config().audio.duration
                except Exception:
                    config_duration = 5.0

                # Handle different audio lengths intelligently
                if duration <= config_duration:
                    # Short audio: use as-is or pad
                    target_length = int(config_duration * sr)
                    if len(y) < target_length:
                        y = np.pad(y, (0, target_length - len(y)))
                elif duration <= 30:
                    # Medium audio: use first segment but inform user
                    target_length = int(config_duration * sr)
                    y = y[:target_length]
                    st.info(f"Audio longer than 30s ({duration:.1f}s); using first 5 seconds for analysis.")
                else:
                    # Long audio: extract multiple segments and average predictions
                    st.info(f"Audio very long ({duration:.1f}s); processing multiple segments for better analysis.")
                    features = self._extract_features_from_long_audio(y, sr, config_duration)
                    return y, sr, features

                # Check if model is multimodal (has multiple inputs)
                is_multimodal = self.model is not None and len(self.model.inputs) > 1

                if is_multimodal:
                    # Extract both MFCC and spectrogram features for multimodal model
                    mfcc_features = self.feature_extractor.extract_mfcc(y, sr)
                    spec_features = self.feature_extractor.extract_spectrogram(y, sr)

                    # Normalize features
                    mfcc_features = self.feature_extractor.normalize_single(mfcc_features, feature_type='mfcc')
                    spec_features = self.feature_extractor.normalize_single(spec_features, feature_type='mel_spectrogram')

                    # Validate shapes
                    expected_mfcc_shape = (13,)  # MFCC features
                    expected_spec_shape = (128, 165, 1)  # Spectrogram features

                    # Fix MFCC shape if needed
                    if mfcc_features.shape != expected_mfcc_shape:
                        if mfcc_features.ndim == 1 and len(mfcc_features) == expected_mfcc_shape[0]:
                            pass  # Already correct
                        else:
                            # Try to reshape or truncate/pad
                            fixed_mfcc = np.zeros(expected_mfcc_shape[0])
                            min_len = min(len(mfcc_features) if mfcc_features.ndim == 1 else mfcc_features.shape[0], expected_mfcc_shape[0])
                            fixed_mfcc[:min_len] = mfcc_features[:min_len] if mfcc_features.ndim == 1 else mfcc_features[:min_len]
                            mfcc_features = fixed_mfcc

                    # Fix spectrogram shape if needed
                    if spec_features.shape != expected_spec_shape:
                        fixed_spec = np.zeros(expected_spec_shape)
                        min_mels = min(spec_features.shape[0], expected_spec_shape[0])
                        min_frames = min(spec_features.shape[1], expected_spec_shape[1])
                        min_channels = min(spec_features.shape[2], expected_spec_shape[2])
                        fixed_spec[:min_mels, :min_frames, :min_channels] = spec_features[:min_mels, :min_frames, :min_channels]
                        spec_features = fixed_spec

                    features = [mfcc_features.reshape(1, -1), spec_features.reshape(1, 128, 165, 1)]
                    st.success(f"✅ Extracted multimodal features: MFCC {mfcc_features.shape}, Spectrogram {spec_features.shape}")

                else:
                    # Standard single-input model (spectrogram only)
                    features = self.feature_extractor.extract_features(y, sr)
                    # Apply normalization learned at training if available
                    features = self.feature_extractor.normalize_single(features, feature_type='mel_spectrogram')

                    # Validate expected shape
                    expected_shape = (1, 128, 165, 1)
                    if features is None or features.shape != expected_shape:
                        st.warning(f"Feature shape mismatch: Expected {expected_shape}, got {None if features is None else features.shape}. Attempting to fix...")
                        fixed = np.zeros(expected_shape)
                        if features is not None:
                            min_batch = min(features.shape[0], expected_shape[0])
                            min_freq = min(features.shape[1], expected_shape[1])
                            min_time = min(features.shape[2], expected_shape[2])
                            min_channel = min(features.shape[3], expected_shape[3])
                            fixed[:min_batch, :min_freq, :min_time, :min_channel] = features[:min_batch, :min_freq, :min_time, :min_channel]
                        features = fixed
                        st.success(f"Fixed feature shape to {features.shape}")

                return y, sr, features
        except Exception as e:
            st.error(f"Error extracting features: {e}")
            import traceback
            st.error(f"Details: {traceback.format_exc()}")
            return None, None, None

    def _extract_features_from_long_audio(self, y, sr, segment_duration=5.0):
        """Extract features from long audio by sampling multiple segments."""
        try:
            segment_samples = int(segment_duration * sr)
            total_samples = len(y)

            # Sample multiple segments from the audio
            num_segments = min(5, total_samples // segment_samples)  # Max 5 segments
            features_list = []

            # Check if model is multimodal
            is_multimodal = self.model is not None and len(self.model.inputs) > 1

            for i in range(num_segments):
                # Sample different parts of the audio
                if num_segments == 1:
                    start_idx = 0
                else:
                    # Distribute segments evenly across the audio
                    start_idx = i * (total_samples - segment_samples) // (num_segments - 1)

                end_idx = start_idx + segment_samples
                segment = y[start_idx:end_idx]

                if is_multimodal:
                    # Extract both MFCC and spectrogram features
                    mfcc_features = self.feature_extractor.extract_mfcc(segment, sr)
                    spec_features = self.feature_extractor.extract_spectrogram(segment, sr)

                    # Normalize features
                    mfcc_features = self.feature_extractor.normalize_single(mfcc_features, feature_type='mfcc')
                    spec_features = self.feature_extractor.normalize_single(spec_features, feature_type='mel_spectrogram')

                    # Fix shapes if needed
                    if mfcc_features.shape != (13,):
                        fixed_mfcc = np.zeros(13)
                        min_len = min(len(mfcc_features), 13)
                        fixed_mfcc[:min_len] = mfcc_features[:min_len]
                        mfcc_features = fixed_mfcc

                    if spec_features.shape != (128, 165, 1):
                        fixed_spec = np.zeros((128, 165, 1))
                        min_mels = min(spec_features.shape[0], 128)
                        min_frames = min(spec_features.shape[1], 165)
                        min_channels = min(spec_features.shape[2], 1)
                        fixed_spec[:min_mels, :min_frames, :min_channels] = spec_features[:min_mels, :min_frames, :min_channels]
                        spec_features = fixed_spec

                    features = [mfcc_features, spec_features]
                else:
                    # Extract spectrogram features only
                    features = self.feature_extractor.extract_features(segment, sr)
                    features = self.feature_extractor.normalize_single(features, feature_type='mel_spectrogram')

                if features is not None:
                    features_list.append(features)

            if features_list:
                if is_multimodal:
                    # Average multimodal features
                    mfcc_list = [f[0] for f in features_list]
                    spec_list = [f[1] for f in features_list]
                    avg_mfcc = np.mean(mfcc_list, axis=0)
                    avg_spec = np.mean(spec_list, axis=0)
                    avg_features = [avg_mfcc.reshape(1, -1), avg_spec.reshape(1, 128, 165, 1)]
                else:
                    # Average spectrogram features
                    avg_features = np.mean(features_list, axis=0)
                st.success(f"✅ Processed {len(features_list)} segments from long audio")
                return avg_features
            else:
                # Fallback to first segment
                y_segment = y[:segment_samples]
                if is_multimodal:
                    mfcc_features = self.feature_extractor.extract_mfcc(y_segment, sr)
                    spec_features = self.feature_extractor.extract_spectrogram(y_segment, sr)
                    mfcc_features = self.feature_extractor.normalize_single(mfcc_features, feature_type='mfcc')
                    spec_features = self.feature_extractor.normalize_single(spec_features, feature_type='mel_spectrogram')
                    features = [mfcc_features.reshape(1, -1), spec_features.reshape(1, 128, 165, 1)]
                else:
                    features = self.feature_extractor.extract_features(y_segment, sr)
                    features = self.feature_extractor.normalize_single(features, feature_type='mel_spectrogram')
                return features

        except Exception as e:
            st.warning(f"Error processing long audio segments: {e}. Using first segment.")
            # Fallback to original method
            y_segment = y[:int(segment_duration * sr)]
            if is_multimodal:
                mfcc_features = self.feature_extractor.extract_mfcc(y_segment, sr)
                spec_features = self.feature_extractor.extract_spectrogram(y_segment, sr)
                mfcc_features = self.feature_extractor.normalize_single(mfcc_features, feature_type='mfcc')
                spec_features = self.feature_extractor.normalize_single(spec_features, feature_type='mel_spectrogram')
                features = [mfcc_features.reshape(1, -1), spec_features.reshape(1, 128, 165, 1)]
            else:
                features = self.feature_extractor.extract_features(y_segment, sr)
                features = self.feature_extractor.normalize_single(features, feature_type='mel_spectrogram')
            return features

    def _check_demo_files_available(self):
        """Check if demo files exist for training."""
        try:
            demo_dir = Path('demo_files')
            if not demo_dir.exists():
                return False

            # Look for audio files
            audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
            audio_files = []
            for ext in audio_extensions:
                audio_files.extend(list(demo_dir.glob(f'*{ext}')))

            return len(audio_files) > 0
        except Exception as e:
            logger.warning(f"Error checking demo files: {e}")
            return False

    def predict_emotion(self, features):
        """Predict emotion from audio features; return graceful defaults if not ready."""
        if not self.loaded or self.model is None:
            logger.error("Prediction requested but model is not loaded yet")
            st.warning("Model not loaded yet. Please wait for training to complete.")
            return "unknown", {}, {}

        try:
            with st.spinner("Predicting emotion..."):
                prediction = self.model.predict(features, verbose=0)
                probs = prediction[0]
                predicted_class = int(np.argmax(probs))
                labels = self.emotion_labels[: len(probs)]
                emotion = labels[predicted_class] if predicted_class < len(labels) else "unknown"
                confidence_scores = {label: float(probs[i]) * 100 for i, label in enumerate(labels)}
                # Uncertainty metrics
                # Normalize probs just in case (model should already output softmax)
                p = np.array(probs, dtype=float)
                p = p / (p.sum() + 1e-12)
                entropy = float(-(p * np.log(p + 1e-12)).sum())  # natural log
                # Convert to bits for interpretability
                entropy_bits = entropy / np.log(2.0)
                # Margin: difference between top1 and top2 probabilities
                if len(p) >= 2:
                    top2 = np.sort(p)[-2:]
                    margin = float(top2[-1] - top2[-2])
                else:
                    margin = 0.0
                uncertainty = {
                    "entropy_nats": entropy,
                    "entropy_bits": entropy_bits,
                    "top2_margin": margin,
                    "num_classes": len(p)
                }
                return emotion, confidence_scores, uncertainty
        except Exception as e:
            st.error(f"Error predicting emotion: {e}")
            return "unknown", {}, {}

    def process_audio(self, audio_file_path):
        """Process audio file and display results"""
        if not self.loaded:
            ready = self.load_model()
            if not ready:
                # Queue this file to auto-process after training completes
                st.session_state['pending_audio_path'] = audio_file_path
                # Show a single global status message via status_container
                if 'training_status' not in st.session_state:
                    st.session_state.training_status = {
                        'type': 'progress',
                        'message': 'Model training is in progress. Your upload will be analyzed automatically when ready.'
                    }
                return

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
            emotion, confidence_scores, uncertainty = self.predict_emotion(features)

            # Display results
            self.display_results(audio_file_path, y, sr, emotion, confidence_scores, uncertainty)

            # Save analysis results for dashboard visualization
            try:
                self.dashboard.save_analysis_result(audio_file_path, emotion, confidence_scores, uncertainty=uncertainty)
            except TypeError:
                # Backwards compatibility if method signature not yet updated
                self.dashboard.save_analysis_result(audio_file_path, emotion, confidence_scores)
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
            import traceback
            st.error(f"Details: {traceback.format_exc()}")

    def display_file_upload(self):
        """Display file upload interface and handle uploaded files"""
        with st.container():
            # Keep high-contrast header without tinted backgrounds
            st.markdown("""
            <div style="margin: 8px 0 12px 0;">
              <h2 style="margin:0; font-weight:600; color:#111827;">Analyze Your Speech</h2>
              <p style="margin:4px 0 0 0; color:#4B5563;">Upload or record audio to detect emotions</p>
            </div>
            """, unsafe_allow_html=True)

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
                    st.markdown("<h3 style='font-weight: 600; color: #111827;'>Upload Audio File</h3>", unsafe_allow_html=True)

                    # Create a modern upload area
                    st.markdown("<div class='uploadArea'>", unsafe_allow_html=True)
                    uploaded_file = st.file_uploader(
                        "Choose an audio file (WAV or MP3)",
                        type=["wav", "mp3"],
                        help="Upload a short audio clip (ideally 5-10 seconds) of someone speaking"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Process uploaded file
                    if uploaded_file is not None:
                        # Validation: size limit
                        MAX_SIZE_MB = 10
                        data = uploaded_file.getvalue()
                        size_mb = len(data) / (1024 * 1024)
                        if size_mb > MAX_SIZE_MB:
                            st.error(f"File is too large ({size_mb:.2f} MB). Limit is {MAX_SIZE_MB} MB.")
                            return
                        # Basic MIME / extension validation
                        import mimetypes
                        mime, _ = mimetypes.guess_type(uploaded_file.name)
                        allowed_mime = {"audio/wav", "audio/x-wav", "audio/mpeg"}
                        if mime and mime not in allowed_mime:
                            st.warning(f"Unexpected MIME type: {mime}. Proceeding cautiously.")
                        # Persist to temp
                        suffix = f".{uploaded_file.name.split('.')[-1].lower()}"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                            try:
                                tmp_file.write(data)
                                temp_path = os.path.abspath(tmp_file.name)
                            except IOError as e:
                                logger.error(f"File write error: {e}")
                                st.error(f"Failed to save uploaded file: {e}")
                                return
                        # Probe duration
                        try:
                            y_probe, sr_probe = librosa.load(temp_path, sr=None, mono=True)
                            duration = len(y_probe) / float(sr_probe)
                            if duration < 0.5:
                                st.warning("Audio clip is very short (<0.5s); prediction may be unreliable.")
                            elif duration > 30:
                                st.info(f"Audio longer than 30s ({duration:.1f}s); only the first segment will be used.")
                        except Exception as e:
                            st.info(f"Could not determine audio duration: {e}")
                        st.success(f"File uploaded: {uploaded_file.name} ({size_mb:.2f} MB)")
                        self.process_audio(temp_path)

                    # Sample button with improved styling
                    st.markdown("<div style='margin-top: 20px; text-align: center;'>", unsafe_allow_html=True)
                    if st.button("🔊 Try a Sample Audio", key="try_sample", use_container_width=True):
                        sample_path = Path("demo_files/happy_sample.wav")
                        if sample_path.exists():
                            self.process_audio(str(sample_path))
                            st.success("Sample audio loaded and analyzed!")
                        else:
                            st.warning("Sample file not found.")
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
                    st.markdown("<h3 style='font-weight: 600; color: #111827;'>Record Your Voice</h3>", unsafe_allow_html=True)
                    st.write("Click the microphone to start/stop recording, then we will analyze your speech.")

                    if audio_recorder is None:
                        st.info("Microphone recording unavailable (install 'audio-recorder-streamlit' to enable).")
                    else:
                        audio_bytes = audio_recorder(text="Start / Stop Recording", recording_color="#ef4444", neutral_color="#4f46e5", icon_name="microphone", icon_size="3x")
                        if audio_bytes:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                                tmp_file.write(audio_bytes)
                                recorded_path = os.path.abspath(tmp_file.name)
                            st.success("Recording captured – analyzing...")
                            self.process_audio(recorded_path)

    def display_results(self, audio_file_path, y, sr, emotion, confidence_scores, uncertainty):
        """Display emotion analysis results"""
        try:
            st.markdown("<hr style='margin: 2rem 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)

            # Configure matplotlib to use default style and color cycle
            plt.style.use('default')
            if not plt.rcParams['axes.prop_cycle']:
                plt.rcParams['axes.prop_cycle'] = plt.cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

            # Create a container for results
            with stylable_container(
                key="results_container",
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
                # Main result header
                st.markdown(f"""
                <div style="text-align: center; margin-bottom: 1.5rem;">
                    <h2 style="font-weight: 600; color: {EMOTION_COLORS.get(emotion, '#333333')}; margin-bottom: 8px;">
                        Detected Emotion: {emotion.capitalize()}
                    </h2>
                    <p style="color: #6B7280; font-size: 1.1rem;">
                        Analysis completed for {os.path.basename(audio_file_path)}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Results in three columns
                col1, col2, col3 = st.columns([1, 1.5, 1])

                # Column 1: Audio waveform
                with col1:
                    st.markdown("<h4 style='font-weight: 600; font-size: 1.1rem; margin-bottom: 1rem;'>Audio Waveform</h4>", unsafe_allow_html=True)

                    # Display audio player
                    st.audio(audio_file_path, format="audio/wav")

                    # Display waveform visualization
                    fig, ax = plt.subplots(figsize=(5, 3))
                    librosa.display.waveshow(y, sr=sr, ax=ax, color='#1f77b4')
                    ax.set_ylabel("Amplitude")
                    ax.set_title("Audio Signal")
                    plt.tight_layout()
                    st.pyplot(fig)

                # Column 2: Emotion confidence & uncertainty
                with col2:
                    st.markdown("<h4 style='font-weight: 600; font-size: 1.1rem; margin-bottom: 1rem;'>Emotion Confidence Scores</h4>", unsafe_allow_html=True)

                    # Create and display interactive bar chart
                    fig = self.create_interactive_visualization(confidence_scores)
                    st.plotly_chart(fig, use_container_width=True)

                    # Display a table with all confidence + uncertainty
                    st.markdown("<h5 style='font-weight: 600; font-size: 0.9rem; margin: 1rem 0 0.5rem 0;'>All Detected Emotions</h5>", unsafe_allow_html=True)

                    # Create a DataFrame for display
                    df = pd.DataFrame({
                        "Emotion": list(confidence_scores.keys()),
                        "Confidence (%)": list(confidence_scores.values())
                    })
                    df = df.sort_values(by="Confidence (%)", ascending=False).reset_index(drop=True)

                    # Format confidence scores to 2 decimal places
                    df["Confidence (%)"] = df["Confidence (%)"].map("{:.2f}%".format)

                    # Display as a modern styled table
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    # Uncertainty metrics panel
                    if uncertainty:
                        ent_bits = uncertainty.get("entropy_bits")
                        margin = uncertainty.get("top2_margin")
                        num_classes = max(1, int(uncertainty.get("num_classes", len(confidence_scores))))
                        # Max entropy in bits for N classes = log2(N)
                        max_entropy_bits = np.log2(num_classes)
                        entropy_norm_pct = (ent_bits / max_entropy_bits * 100.0) if max_entropy_bits > 0 else 0.0
                        st.markdown("<h5 style='font-weight: 600; font-size: 0.9rem; margin: 1rem 0 0.5rem 0;'>Uncertainty Metrics</h5>", unsafe_allow_html=True)
                        st.caption(f"Entropy: {ent_bits:.3f} bits (of {max_entropy_bits:.3f}), Normalized: {entropy_norm_pct:.1f}% | Top-2 Margin: {margin:.3f}")

                # Column 3: Gauge chart and spectrogram
                with col3:
                    st.markdown("<h4 style='font-weight: 600; font-size: 1.1rem; margin-bottom: 1rem;'>Confidence Meter</h4>", unsafe_allow_html=True)

                    # Create gauge chart for the primary emotion
                    primary_confidence = confidence_scores.get(emotion, 0)
                    fig = self.create_gauge_chart(primary_confidence, emotion)
                    st.plotly_chart(fig, use_container_width=True)

                    # Show model-level accuracy if available
                    try:
                        metrics = self.model_manager.get_model_evaluation_report(model_path=self.model_path)
                    except Exception:
                        metrics = None
                    if metrics and isinstance(metrics, dict) and metrics.get('accuracy') is not None:
                        acc = float(metrics['accuracy'])
                        if acc <= 1.0:
                            acc *= 100.0
                        st.markdown("<h5 style='font-weight: 600; font-size: 0.9rem; margin: 1rem 0 0.5rem 0;'>Model Accuracy (test)</h5>", unsafe_allow_html=True)
                        st.markdown(f"<div class='accuracy-meter'><div class='accuracy-value' style='width: {acc:.1f}%'></div></div>", unsafe_allow_html=True)
                        st.caption(f"Overall test accuracy: {acc:.1f}%")

                    # Display spectrogram visualization
                    st.markdown("<h5 style='font-weight: 600; font-size: 0.9rem; margin: 1rem 0 0.5rem 0;'>Mel Spectrogram</h5>", unsafe_allow_html=True)

                    fig, ax = plt.subplots(figsize=(5, 3))
                    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
                    S_dB = librosa.power_to_db(mel_spec, ref=np.max)
                    _ = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax)
                    ax.set_title("Mel Spectrogram")
                    plt.tight_layout()
                    st.pyplot(fig)

                # Historical comparison section
                st.markdown("<h4 style='font-weight: 600; font-size: 1.1rem; margin: 1.5rem 0 1rem 0;'>Emotion Analysis Insights</h4>", unsafe_allow_html=True)

                insight_cols = st.columns(3)

                # Insight 1: Primary emotion
                with insight_cols[0]:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {EMOTION_COLORS.get(emotion, '#607D8B')}22 0%, {EMOTION_COLORS.get(emotion, '#607D8B')}11 100%); padding: 16px; border-radius: 12px; height: 100%;">
                        <h5 style="color: {EMOTION_COLORS.get(emotion, '#333333')}; font-weight: 600; font-size: 1rem; margin-bottom: 8px;">Primary Emotion</h5>
                        <p style="font-size: 2rem; margin: 8px 0; font-weight: 700; color: {EMOTION_COLORS.get(emotion, '#333333')};">{emotion.capitalize()}</p>
                        <p style="color: #4B5563; font-size: 0.9rem;">Dominant emotional tone detected in the audio</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Insight 2: Confidence level
                with insight_cols[1]:
                    # Determine confidence level text
                    if primary_confidence > 80:
                        confidence_text = "High Confidence"
                        confidence_description = "The model is very confident in this emotion classification"
                    elif primary_confidence > 50:
                        confidence_text = "Moderate Confidence"
                        confidence_description = "The model has a moderate level of certainty in this classification"
                    else:
                        confidence_text = "Low Confidence"
                        confidence_description = "The emotion may be subtle or mixed with other emotions"

                    st.markdown(f"""
                    <div style="background-color: #F8FAFC; padding: 16px; border-radius: 12px; height: 100%; border: 1px solid #E2E8F0;">
                        <h5 style="color: #1E293B; font-weight: 600; font-size: 1rem; margin-bottom: 8px;">Confidence Level</h5>
                        <p style="font-size: 2rem; margin: 8px 0; font-weight: 700; color: #1E293B;">{confidence_text}</p>
                        <p style="color: #4B5563; font-size: 0.9rem;">{confidence_description}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Insight 3: Secondary emotion (if any)
                with insight_cols[2]:
                    # Find secondary emotion (second highest confidence)
                    emotions_sorted = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True) if confidence_scores else []
                    if len(emotions_sorted) > 1:
                        secondary_emotion = emotions_sorted[1][0]
                        secondary_confidence = emotions_sorted[1][1]

                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {EMOTION_COLORS.get(secondary_emotion, '#607D8B')}22 0%, {EMOTION_COLORS.get(secondary_emotion, '#607D8B')}11 100%); padding: 16px; border-radius: 12px; height: 100%;">
                            <h5 style="color: {EMOTION_COLORS.get(secondary_emotion, '#333333')}; font-weight: 600; font-size: 1rem; margin-bottom: 8px;">Secondary Emotion</h5>
                            <p style="font-size: 2rem; margin: 8px 0; font-weight: 700; color: {EMOTION_COLORS.get(secondary_emotion, '#333333')};">{secondary_emotion.capitalize()}</p>
                            <p style="color: #4B5563; font-size: 0.9rem;">Also detected with {secondary_confidence:.1f}% confidence</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background-color: #F8FAFC; padding: 16px; border-radius: 12px; height: 100%; border: 1px solid #E2E8F0;">
                            <h5 style="color: #1E293B; font-weight: 600; font-size: 1rem; margin-bottom: 8px;">Secondary Emotion</h5>
                            <p style="font-size: 1.2rem; margin: 8px 0; font-weight: 500; color: #6B7280;">None Detected</p>
                            <p style="color: #4B5563; font-size: 0.9rem;">No significant secondary emotion was found</p>
                        </div>
                        """, unsafe_allow_html=True)

                # Call-to-action to view dashboard
                st.markdown("""
                <div style="margin-top: 24px; text-align: center;">
                    <p style="color: #4B5563; font-size: 0.95rem; margin-bottom: 12px;">Want to see trends and patterns across all your analyses?</p>
                </div>
                """, unsafe_allow_html=True)

                dashboard_col1, dashboard_col2, _dashboard_col3 = st.columns([1, 1, 1])
                with dashboard_col2:
                    if st.button("View Visualization Dashboard", use_container_width=True):
                        # Use query parameters to navigate to dashboard
                        st.experimental_set_query_params(page="visualization_dashboard")
                        st.rerun()

        except Exception as e:
            st.error(f"Error displaying results: {e}")
            import traceback
            st.error(f"Details: {traceback.format_exc()}")

    def display_demo_section(self):
        """Display the demo section with sample audio files"""
        with st.container():
            st.markdown("""
            <div style="margin: 8px 0 12px 0;">
              <h2 style="margin:0; font-weight:600; color:#111827;">Example Audio Samples</h2>
              <p style="margin:4px 0 0 0; color:#4B5563;">Listen to and analyze audio samples with different emotions</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <p style="color: #4B5563; font-size: 1.05rem; margin-bottom: 24px; max-width: 800px;">
                These samples demonstrate different emotional states in speech. Listen to the audio and 
                click the analyze button to see how our AI classifies each emotion.
            </p>
            """, unsafe_allow_html=True)

            # Create columns for sample cards
            col1, col2, col3 = st.columns(3)

            # Define emotion samples
            emotion_samples = [
                {
                    "emotion": "happy",
                    "icon": "😄",
                    "color": "#FFB300",
                    "gradient": "linear-gradient(135deg, #FFF8E1 0%, #FFECB3 100%)",
                    "description": "Example of a joyful voice with higher pitch and energetic tone."
                },
                {
                    "emotion": "angry",
                    "icon": "😠",
                    "color": "#D32F2F",
                    "gradient": "linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%)",
                    "description": "Example of an aggressive voice with intense tone and sharp articulation."
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

    def display_tensorboard_launcher(self):
        """Display TensorBoard launcher section"""
        with st.container():
            st.markdown("""
            <div style=\"margin: 8px 0 12px 0;\"> 
              <h2 style=\"margin:0; font-weight:600; color:#111827;\">TensorBoard Visualization</h2>
              <p style=\"margin:4px 0 0 0; color:#4B5563;\">Launch TensorBoard to visualize model training metrics</p>
            </div>
            """, unsafe_allow_html=True)

            # Create a modern UI container for TensorBoard launcher
            with stylable_container(
                key="tb_container",
                css_styles="""
                    {
                        background-color: white;
                        border-radius: 16px;
                        padding: 28px;
                        margin-top: 16px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 20px 25px -5px rgba(0,0,0,0.05);
                        border: 1px solid rgba(226, 232, 240, 0.8);
                    }
                """
            ):
                st.markdown("""
                <h3 style="font-weight: 600; color: #4F46E5; margin-bottom: 16px;">TensorBoard</h3>
                <p style="color: #4B5563; margin-bottom: 20px;">
                    TensorBoard provides visualizations of model training metrics, helping you understand 
                    the training process and model performance.
                </p>
                """, unsafe_allow_html=True)

                # Check if TensorBoard is already running
                if not st.session_state.get('tensorboard_running', False):
                    # Not running, show launcher
                    st.markdown("""
                    <div style="background-color: #F8FAFC; padding: 20px; border-radius: 12px; margin-bottom: 24px;">
                        <h4 style="color: #1E293B; font-weight: 600; font-size: 1.1rem; margin-bottom: 12px;">Launch TensorBoard</h4>
                        <p style="color: #4B5563; margin-bottom: 16px;">
                            Select a log directory containing TensorFlow training logs, then click "Start TensorBoard" to launch the visualization server.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Default logs directory
                    default_logs_dir = "logs"
                    logs_dirs = [d for d in os.listdir(default_logs_dir) if os.path.isdir(os.path.join(default_logs_dir, d))] if os.path.exists(default_logs_dir) else []

                    if logs_dirs:
                        # Create a more descriptive format for the dropdown
                        log_options = []
                        for log_dir in logs_dirs:
                            # Try to extract date from folder name if it follows a pattern like run_20250504_154714
                            if log_dir.startswith("run_") and len(log_dir) > 12:
                                try:
                                    date_str = log_dir[4:12]  # Extract 20250504
                                    time_str = log_dir[13:19] if len(log_dir) > 18 else ""  # Extract 154714

                                    # Format as YYYY-MM-DD HH:MM:SS
                                    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                                    formatted_time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}" if time_str else ""

                                    display_name = f"{formatted_date} {formatted_time} ({log_dir})"
                                except Exception:
                                    display_name = log_dir
                            else:
                                display_name = log_dir

                            log_options.append({"label": display_name, "value": log_dir})

                        # Sort log options by date (most recent first)
                        log_options.sort(key=lambda x: x["value"], reverse=True)

                        # Get log dir path from dropdown
                        selected_log = st.selectbox(
                            "Select a training log directory:", 
                            options=[opt["value"] for opt in log_options],
                            format_func=lambda x: next((opt["label"] for opt in log_options if opt["value"] == x), x)
                        )

                        logs_dir = os.path.join(default_logs_dir, selected_log)

                        # Add a port selection slider
                        port = st.slider("TensorBoard Port", min_value=6006, max_value=6016, value=6006, step=1)

                        # Launch button
                        if st.button("Start TensorBoard Server", use_container_width=True):
                            if os.path.exists(logs_dir):
                                try:
                                    # Check if port is available
                                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                    port_in_use = False
                                    try:
                                        s.bind(("127.0.0.1", port))
                                    except Exception:
                                        port_in_use = True
                                    finally:
                                        s.close()

                                    if port_in_use:
                                        st.warning(f"Port {port} is already in use. TensorBoard might already be running.")

                                    # Launch TensorBoard as a subprocess
                                    cmd = f"tensorboard --logdir={logs_dir} --port={port}"

                                    try:
                                        self.tensorboard_process = _subprocess.Popen(
                                            cmd, 
                                            shell=True,
                                            stdout=_subprocess.PIPE,
                                            stderr=_subprocess.PIPE,
                                            text=True
                                        )
                                        self.tensorboard_port = port

                                        # Update session state
                                        st.session_state['tensorboard_running'] = True
                                        st.success(f"TensorBoard started on http://localhost:{port}")

                                        # Wait a moment for TensorBoard to start
                                        time.sleep(2)

                                        # Try to automatically open TensorBoard in a browser
                                        webbrowser.open(f"http://localhost:{port}")

                                        # Rerun to show the TensorBoard iframe
                                        st.rerun()

                                    except Exception as e:
                                        st.error(f"Error starting TensorBoard: {e}")
                                except Exception as e:
                                    st.error(f"Error: {e}")
                            else:
                                st.error(f"Log directory {logs_dir} does not exist.")
                    else:
                        st.info("No training log directories found. Check the 'logs' folder or run model training first.")
                else:
                    # TensorBoard is running, show status and control
                    port = getattr(self, 'tensorboard_port', 6006)

                    st.markdown(f"""
                    <div style="background-color: #f0e6ff; border-radius: 12px; padding: 20px; margin-bottom: 24px; border: 1px solid #d0c0ff;">
                        <h4 style="color: #6B21A8; font-weight: 600; font-size: 1.1rem; margin-bottom: 12px;">
                            <span style="margin-right: 8px;">🚀</span> TensorBoard is Running
                        </h4>
                        <p style="margin-bottom: 16px;">
                            TensorBoard is currently running on port {port}. You can access it using the link below or by opening a browser with the URL.
                        </p>
                        <p style="margin-bottom: 16px; font-weight: 500;">
                            <a href="http://localhost:{port}" target="_blank">http://localhost:{port}</a>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button("Stop TensorBoard Server", use_container_width=True):
                        if self.tensorboard_process:
                            self.tensorboard_process.terminate()
                            self.tensorboard_process = None
                            st.session_state['tensorboard_running'] = False
                            st.success("TensorBoard server stopped successfully")
                            st.rerun()

    def display_about_section(self):
        """Display about section with project information"""
        with st.container():
            st.markdown("""
            <div style=\"margin: 8px 0 12px 0;\"> 
              <h2 style=\"margin:0; font-weight:600; color:#111827;\">About Speech Emotion Analyzer</h2>
              <p style=\"margin:4px 0 0 0; color:#4B5563;\">Learn about the project, methodology, and technology</p>
            </div>
            """, unsafe_allow_html=True)

            # Create columns for about section
            col1, col2 = st.columns([2, 1])

            with col1:
                with stylable_container(
                    key="about_container",
                    css_styles="""
                        {
                            background-color: white;
                            border-radius: 16px;
                            padding: 28px;
                            margin-top: 16px;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 20px 25px -5px rgba(0,0,0,0.05);
                            border: 1px solid rgba(226, 232, 240, 0.8);
                        }
                    """
                ):
                    st.markdown("""
                    <h3 style="font-weight: 600; color: #4F46E5; margin-bottom: 16px;">Project Overview</h3>
                    
                    <p style="color: #1F2937; line-height: 1.6; margin-bottom: 16px;">
                        The Speech Emotion Analyzer is an AI-powered application that recognizes emotions in speech using 
                        deep learning technology. It extracts acoustic features from audio and classifies the emotional 
                        content using a Convolutional Neural Network (CNN) trained on emotional speech datasets.
                    </p>
                    
                    <h4 style="font-weight: 600; color: #1F2937; margin: 24px 0 12px 0; font-size: 1.1rem;">How It Works</h4>
                    
                    <ol style="color: #4B5563; line-height: 1.6; margin-bottom: 16px; padding-left: 20px;">
                        <li><strong>Audio Input:</strong> The system accepts audio files (WAV or MP3) containing speech.</li>
                        <li><strong>Feature Extraction:</strong> Acoustic features like Mel spectrograms are extracted from the audio.</li>
                        <li><strong>CNN Processing:</strong> A deep neural network analyzes the features to detect emotional patterns.</li>
                        <li><strong>Classification:</strong> The model classifies the speech into one of 8 emotional categories.</li>
                        <li><strong>Visualization:</strong> Results are displayed with confidence scores and visual analytics.</li>
                    </ol>
                    
                    <h4 style="font-weight: 600; color: #1F2937; margin: 24px 0 12px 0; font-size: 1.1rem;">Technologies Used</h4>
                    
                    <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px;">
                        <span style="background-color: #EFF6FF; color: #1E40AF; padding: 6px 12px; border-radius: 100px; font-size: 0.9rem;">TensorFlow</span>
                        <span style="background-color: #ECFDF5; color: #065F46; padding: 6px 12px; border-radius: 100px; font-size: 0.9rem;">Python</span>
                        <span style="background-color: #F5F3FF; color: #5B21B6; padding: 6px 12px; border-radius: 100px; font-size: 0.9rem;">Librosa</span>
                        <span style="background-color: #FEF3F2; color: #B42318; padding: 6px 12px; border-radius: 100px; font-size: 0.9rem;">NumPy</span>
                        <span style="background-color: #F8FAFC; color: #0F172A; padding: 6px 12px; border-radius: 100px; font-size: 0.9rem;">Pandas</span>
                        <span style="background-color: #F0FDF4; color: #166534; padding: 6px 12px; border-radius: 100px; font-size: 0.9rem;">Streamlit</span>
                        <span style="background-color: #FDF4FF; color: #86198F; padding: 6px 12px; border-radius: 100px; font-size: 0.9rem;">Plotly</span>
                    </div>
                    
                    <h4 style="font-weight: 600; color: #1F2937; margin: 24px 0 12px 0; font-size: 1.1rem;">Model Architecture</h4>
                    
                    <p style="color: #4B5563; line-height: 1.6; margin-bottom: 16px;">
                        The emotion recognition model uses a Convolutional Neural Network (CNN) architecture with multiple 
                        convolutional and pooling layers followed by dense layers. The model was trained on the RAVDESS 
                        (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset, which contains recordings of 
                        professional actors expressing different emotions.
                    </p>
                    
                    <h4 style="font-weight: 600; color: #1F2937; margin: 24px 0 12px 0; font-size: 1.1rem;">Limitations</h4>
                    
                    <ul style="color: #4B5563; line-height: 1.6; margin-bottom: 16px; padding-left: 20px;">
                        <li>The model performs best on clear audio with minimal background noise</li>
                        <li>Short clips (5-10 seconds) work better than longer recordings</li>
                        <li>Performance may vary across different accents and languages</li>
                        <li>Emotional expressions can be culturally dependent</li>
                    </ul>
                    """, unsafe_allow_html=True)

            with col2:
                with stylable_container(
                    key="sidebar_about",
                    css_styles="""
                        {
                            background-color: white;
                            border-radius: 16px;
                            padding: 24px;
                            margin-top: 16px;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 20px 25px -5px rgba(0,0,0,0.05);
                            border: 1px solid rgba(226, 232, 240, 0.8);
                        }
                    """
                ):
                    st.markdown("""
                    <h4 style="font-weight: 600; color: #4F46E5; margin-bottom: 16px; font-size: 1.1rem;">Recognized Emotions</h4>
                    """, unsafe_allow_html=True)
                    
                    # Display recognized emotions with icons
                    emotions = [
                        {"name": "Neutral", "icon": "😐", "color": "#607D8B", "desc": "Lack of emotional expression"},
                        {"name": "Calm", "icon": "😌", "color": "#1E88E5", "desc": "Relaxed, peaceful tone"},
                        {"name": "Happy", "icon": "😄", "color": "#FFB300", "desc": "Joyful, excited expression"},
                        {"name": "Sad", "icon": "😢", "color": "#5E35B1", "desc": "Melancholic, downcast tone"},
                        {"name": "Angry", "icon": "😠", "color": "#D32F2F", "desc": "Irritated, hostile expression"},
                        {"name": "Fearful", "icon": "😨", "color": "#7CB342", "desc": "Anxious, threatened tone"},
                        {"name": "Disgust", "icon": "🤢", "color": "#00897B", "desc": "Averse, repulsed expression"},
                        {"name": "Surprised", "icon": "😲", "color": "#F06292", "desc": "Astonished, startled tone"}
                    ]
                    
                    for emotion in emotions:
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; margin-bottom: 12px; padding: 8px; border-radius: 8px; background-color: {emotion['color']}15;">
                            <div style="font-size: 1.5rem; margin-right: 12px; min-width: 36px; text-align: center;">{emotion['icon']}</div>
                            <div>
                                <div style="font-weight: 500; color: {emotion['color']};">{emotion['name']}</div>
                                <div style="font-size: 0.8rem; color: #6B7280;">{emotion['desc']}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Additional information cards
                with stylable_container(
                    key="dataset_info",
                    css_styles="""
                        {
                            background-color: #F8FAFC;
                            border-radius: 16px;
                            padding: 20px;
                            margin-top: 16px;
                            border: 1px solid #E2E8F0;
                        }
                    """
                ):
                    st.markdown("""
                    <h4 style="font-weight: 600; color: #1E293B; margin-bottom: 12px; font-size: 1rem;">Training Dataset</h4>
                    <p style="color: #4B5563; font-size: 0.9rem; line-height: 1.5;">
                        The model was trained on the RAVDESS dataset, featuring professional actors expressing emotions in standardized statements.
                    </p>
                    <div style="font-size: 0.85rem; color: #64748B; margin-top: 8px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                            <span>Total samples:</span>
                            <span style="font-weight: 500;">1,440+</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                            <span>Professional actors:</span>
                            <span style="font-weight: 500;">24</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span>Gender balance:</span>
                            <span style="font-weight: 500;">50% male, 50% female</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    def display_model_management_section(self):
        """Display model management and settings section"""
        with st.container():
            # Model Management Section
            with stylable_container(
                key="model_management_container",
                css_styles="""
                    {
                        background: rgba(255, 255, 255, 0.95);
                        backdrop-filter: blur(20px);
                        -webkit-backdrop-filter: blur(20px);
                        border-radius: 20px;
                        padding: 28px;
                        margin-bottom: 24px;
                        border: 1px solid rgba(255, 255, 255, 0.3);
                        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                    }
                """
            ):
                st.markdown("<h3 style='font-weight: 600; color: #4F46E5; margin-bottom: 16px;'>🧠 Model Management</h3>", unsafe_allow_html=True)

                # Current Model Status
                col1, col2 = st.columns([1, 1])

                with col1:
                    if self.loaded and self.model is not None:
                        st.success("✅ Model Loaded")
                        st.info(f"**Path:** {self.model_path}")
                        try:
                            model_info = self.model_manager.get_model_by_path(self.model_path)
                            if model_info:
                                st.info(f"**Type:** {model_info.get('type', 'Unknown').upper()}")
                                st.info(f"**Created:** {model_info.get('created', 'Unknown')}")
                                if 'metrics' in model_info and 'accuracy' in model_info['metrics']:
                                    accuracy = model_info['metrics']['accuracy']
                                    if accuracy <= 1.0:
                                        accuracy *= 100
                                    st.metric("Accuracy", f"{accuracy:.1f}%")
                        except Exception:
                            pass
                    else:
                        st.warning("⚠️ No Model Loaded")
                        if st.button("🔄 Load Available Models", use_container_width=True):
                            self.load_model()
                            st.rerun()

                with col2:
                    # Available Models
                    try:
                        models = self.model_manager.get_models()
                        st.info(f"**Available Models:** {len(models)}")

                        if models:
                            # Model selection dropdown
                            model_options = []
                            for model in models:
                                accuracy = "Unknown"
                                if 'metrics' in model and 'accuracy' in model['metrics']:
                                    acc = model['metrics']['accuracy']
                                    if acc <= 1.0:
                                        acc *= 100
                                    accuracy = ".1f"

                                option_text = f"{model.get('type', 'Unknown').upper()} - {model.get('created', 'Unknown')[:16]} (Acc: {accuracy})"
                                model_options.append((model['id'], option_text))

                            if model_options:
                                selected_model = st.selectbox(
                                    "Switch to Model:",
                                    options=[opt[0] for opt in model_options],
                                    format_func=lambda x: next((opt[1] for opt in model_options if opt[0] == x), x)
                                )

                                if st.button("🔄 Switch Model", use_container_width=True):
                                    try:
                                        model_info = self.model_manager.get_model_by_id(selected_model)
                                        if model_info and model_info.get('path'):
                                            self.model = tf.keras.models.load_model(model_info['path'])
                                            self.model_path = model_info['path']
                                            self.loaded = True
                                            st.success(f"✅ Switched to model: {selected_model}")
                                            st.rerun()
                                        else:
                                            st.error("❌ Could not load selected model")
                                    except Exception as e:
                                        st.error(f"❌ Error switching model: {e}")

                    except Exception as e:
                        st.error(f"Error loading model list: {e}")

            # Training Section
            with stylable_container(
                key="training_container",
                css_styles="""
                    {
                        background: rgba(255, 255, 255, 0.95);
                        backdrop-filter: blur(20px);
                        -webkit-backdrop-filter: blur(20px);
                        border-radius: 20px;
                        padding: 28px;
                        margin-bottom: 24px;
                        border: 1px solid rgba(255, 255, 255, 0.3);
                        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                    }
                """
            ):
                st.markdown("<h3 style='font-weight: 600; color: #4F46E5; margin-bottom: 16px;'>🚀 Training Options</h3>", unsafe_allow_html=True)

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.markdown("**Manual Training**")
                    st.markdown("Train a new model using the main training pipeline.")
                    if st.button("🎯 Start Full Training", use_container_width=True, type="primary"):
                        if not self.training_in_progress:
                            self.train_model_automatically()
                        else:
                            st.warning("Training is already in progress")

                with col2:
                    st.markdown("**Quick Training**")
                    st.markdown("Train a basic model using demo files (faster).")
                    if st.button("⚡ Quick Train", use_container_width=True):
                        if not self.training_in_progress:
                            try:
                                import subprocess
                                st.info("Starting quick training...")
                                result = subprocess.run([
                                    sys.executable, 'auto_train.py'
                                ], capture_output=True, text=True, timeout=300)

                                if result.returncode == 0:
                                    st.success("✅ Quick training completed!")
                                    self.load_model()
                                    st.rerun()
                                else:
                                    st.error(f"❌ Quick training failed: {result.stderr}")
                            except Exception as e:
                                st.error(f"❌ Error during quick training: {e}")
                        else:
                            st.warning("Training is already in progress")

            # System Information
            with stylable_container(
                key="system_info_container",
                css_styles="""
                    {
                        background: rgba(255, 255, 255, 0.95);
                        backdrop-filter: blur(20px);
                        -webkit-backdrop-filter: blur(20px);
                        border-radius: 20px;
                        padding: 28px;
                        border: 1px solid rgba(255, 255, 255, 0.3);
                        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                    }
                """
            ):
                st.markdown("<h3 style='font-weight: 600; color: #4F46E5; margin-bottom: 16px;'>📊 System Information</h3>", unsafe_allow_html=True)

                col1, col2, col3 = st.columns([1, 1, 1])

                with col1:
                    st.metric("TensorFlow Available", "✅" if tensorflow_available else "❌")
                    if tensorflow_available:
                        try:
                            st.metric("TensorFlow Version", tf.__version__)
                        except:
                            st.metric("TensorFlow Version", "Unknown")

                with col2:
                    # Check demo files
                    demo_available = self._check_demo_files_available()
                    st.metric("Demo Files Available", "✅" if demo_available else "❌")

                    # Check models directory
                    models_dir = Path('models')
                    models_available = models_dir.exists() and len(list(models_dir.glob('*.keras'))) > 0
                    st.metric("Models Available", "✅" if models_available else "❌")

                with col3:
                    # Training status
                    training_status = "✅ Ready" if not self.training_in_progress else "🔄 In Progress"
                    st.metric("Training Status", training_status)

                    # Model loaded status
                    model_status = "✅ Loaded" if self.loaded and self.model else "❌ Not Loaded"
                    st.metric("Model Status", model_status)

    def display_settings_section(self):
        """Legacy settings section - now redirects to model management"""
        self.display_model_management_section()
        
    def run(self):
        """Main method to run the Streamlit application"""
        # Inject CSS once per run
        self._inject_css()
        # If a file was queued during training and model is now ready, process it automatically
        if self.loaded and st.session_state.get('pending_audio_path'):
            pending_path = st.session_state.get('pending_audio_path')
            if pending_path and os.path.exists(pending_path):
                st.info("Processing your uploaded file now that the model is ready...")
                # Clear the pending state first to avoid any recursion on rerun
                st.session_state.pop('pending_audio_path', None)
                self.process_audio(pending_path)
        # Display modern app header with gradient text and design
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem; padding: 1.5rem 0;">
            <h1 class='main-header'>Speech Emotion Analyzer</h1>
            <p style="text-align: center; font-size: 1.1rem; color: #374151; max-width: 700px; margin: 0 auto 1.25rem auto; line-height: 1.6;">
                Detect emotions in speech using AI-powered deep learning technology
            </p>
            <div style="display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; margin-top: 8px;">
                <span style="background-color: #EEF2FF; color: #4338CA; padding: 4px 12px; border-radius: 100px; font-size: 0.85rem; font-weight: 500;">Deep Learning</span>
                <span style="background-color: #E0F2FE; color: #0369A1; padding: 4px 12px; border-radius: 100px; font-size: 0.85rem; font-weight: 500;">CNN Architecture</span>
                <span style="background-color: #DCFCE7; color: #15803D; padding: 4px 12px; border-radius: 100px; font-size: 0.85rem; font-weight: 500;">Real-time Analysis</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar navigation
        with st.sidebar:
            try:
                # Check if visualization image exists before trying to display it
                if os.path.exists("results/visualizations/enhanced_confusion_matrix.png"):
                    st.image("results/visualizations/enhanced_confusion_matrix.png", caption="Emotion Classification Matrix", use_container_width=True)
                else:
                    st.info("Visualization image not found. This won't affect the app's functionality.")
            except Exception:
                st.info("Could not load visualization image. The app will still function normally.")
            
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
            st.markdown("""
            <div style=\"margin: 8px 0 12px 0;\"> 
              <h2 style=\"margin:0; font-weight:600; color:#111827;\">Emotion Analytics Dashboard</h2>
              <p style=\"margin:4px 0 0 0; color:#4B5563;\">Visualize and analyze your emotion detection results</p>
            </div>
            """, unsafe_allow_html=True)
            self.dashboard.display_dashboard()
            
        elif selected == "View Examples":
            self.display_demo_section()
            
        elif selected == "TensorBoard":
            self.display_tensorboard_launcher()
            
        elif selected == "About":
            self.display_about_section()
            
        elif selected == "Settings":
            st.markdown("""
            <div style=\"margin: 8px 0 12px 0;\">
              <h2 style=\"margin:0; font-weight:600; color:#111827;\">Settings & Model Management</h2>
              <p style=\"margin:4px 0 0 0; color:#4B5563;\">Configure application settings and manage trained models</p>
            </div>
            """, unsafe_allow_html=True)
            self.display_model_management_section()
        
        # Footer with credits and additional information
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

    def train_model_automatically(self):
        """Run training in a background thread with file lock to avoid overlap."""
        models_dir = Path('models')
        models_dir.mkdir(parents=True, exist_ok=True)
        lock_path = models_dir / 'training.lock'

        if self.training_in_progress:
            logger.info("Training already in progress, skipping...")
            return False

        try:
            lock_file = open(lock_path, 'w')
            try:
                portalocker.lock(lock_file, portalocker.LOCK_EX | portalocker.LOCK_NB)
                logger.info("Acquired training lock successfully")
            except portalocker.LockException:
                self.training_in_progress = True
                st.info("⏳ Model training already in progress. Your file will be analyzed when ready.")
                logger.info("Training lock already held by another process")
                return False
        except Exception as e:
            logger.warning(f"Could not acquire training lock: {e}")
            # If locking fails, proceed but still try training (best effort)
            lock_file = None

        self.training_in_progress = True
        st.info("🚀 Starting automatic model training. This may take a few minutes...")
        progress_placeholder = st.empty()

        def _run():
            start = time.time()
            model_path = None
            try:
                logger.info("Starting model training with CNN and mel_spectrogram features...")
                args = TrainArgs(model_type='cnn', feature_type='mel_spectrogram')
                model_path, metrics = train_model(args)

                if model_path and os.path.exists(model_path):
                    elapsed = time.time() - start
                    st.session_state.training_status = {
                        'type': 'success',
                        'message': f'Training completed in {int(elapsed//60):02d}:{int(elapsed%60):02d}. Model saved to {model_path}'
                    }
                    logger.info(f"Training completed successfully. Model saved to: {model_path}")
                else:
                    # Try automatic training as fallback
                    logger.warning("Main training failed, attempting automatic training...")
                    st.session_state.training_status = {
                        'type': 'progress',
                        'message': 'Main training failed, trying automatic training with demo files...'
                    }

                    try:
                        import subprocess
                        result = subprocess.run([
                            sys.executable, 'auto_train.py'
                        ], capture_output=True, text=True, timeout=600)  # 10 minute timeout

                        if result.returncode == 0:
                            # Try to find the auto-trained model
                            auto_model_path = 'models/cnn_emotion_model_auto.keras'
                            if os.path.exists(auto_model_path):
                                model_path = auto_model_path
                                elapsed = time.time() - start
                                st.session_state.training_status = {
                                    'type': 'success',
                                    'message': f'Automatic training completed in {int(elapsed//60):02d}:{int(elapsed%60):02d}. Model saved!'
                                }
                                logger.info("Automatic training completed successfully")
                            else:
                                raise Exception("Auto training completed but model file not found")
                        else:
                            raise Exception(f"Auto training failed: {result.stderr}")

                    except Exception as auto_e:
                        logger.error(f"Automatic training also failed: {auto_e}")
                        elapsed = time.time() - start
                        st.session_state.training_status = {
                            'type': 'error',
                            'message': f'Both training methods failed after {int(elapsed//60):02d}:{int(elapsed%60):02d}'
                        }
                        return

                # Load the newly trained model
                if model_path:
                    success = self.load_trained_model()
                    if success:
                        st.success("✅ Model loaded and ready for predictions!")
                        logger.info("Newly trained model loaded successfully")
                    else:
                        st.warning("⚠️ Model was trained but couldn't be loaded. Please refresh the page.")
                        logger.warning("Failed to load newly trained model")

            except Exception as e:
                elapsed = time.time() - start
                error_msg = f'Training failed after {int(elapsed//60):02d}:{int(elapsed%60):02d}: {str(e)}'
                st.session_state.training_status = {'type': 'error', 'message': error_msg}
                logger.error(f'Training error: {e}')
                import traceback
                logger.error(f'Training traceback: {traceback.format_exc()}')
            finally:
                self.training_in_progress = False
                if lock_file:
                    try:
                        portalocker.unlock(lock_file)
                        lock_file.close()
                        lock_path.unlink(missing_ok=True)
                        logger.info("Released training lock")
                    except Exception as e:
                        logger.warning(f"Error releasing training lock: {e}")

        threading.Thread(target=_run, daemon=True).start()
        return True
    
    def _monitor_training_process(self, progress_placeholder):
        """Monitor the training process and update the UI with progress"""
        try:
            start_time = time.time()
            
            while self.training_process.poll() is None:
                elapsed = time.time() - start_time
                mins, secs = divmod(int(elapsed), 60)
                
                # Update progress message every few seconds
                st.session_state.training_status = {
                'type': 'progress',
                'message': f'Training a new emotion recognition model. Time elapsed: {mins:02d}:{secs:02d}'
            }
                
                time.sleep(2)
            
            # Process completed
            return_code = self.training_process.returncode
            stdout, stderr = self.training_process.communicate()
            
            if return_code == 0:
                # Training successful
                elapsed = time.time() - start_time
                mins, secs = divmod(int(elapsed), 60)
                st.session_state.training_status = {
                    'type': 'success',
                    'message': f'Training completed successfully in {mins:02d}:{secs:02d}. Attempting to load the new model...'
                }
                
                # Try to load the newly trained model
                self.load_trained_model()
            else:
                # Training failed
                st.session_state.training_status = {
                    'type': 'error',
                    'message': 'There was an error during model training. Please check the logs for details.'
                }
                logger.error(f"Training process failed with return code {return_code}")
                logger.error(f"Error output: {stderr}")
            
            # Reset training status and remove lock
            self.training_in_progress = False
            try:
                (Path('models') / 'training.lock').unlink(missing_ok=True)
            except Exception:
                pass
            
        except Exception as e:
            logger.error(f"Error monitoring training process: {e}")
            self.training_in_progress = False
            progress_placeholder.markdown(f"""
            <div style='padding: 10px; border-radius: 5px; background-color: #f8d7da; border: 1px solid #dc3545; margin: 10px 0;'>
                <h4 style='margin: 0; color: #721c24;'>❌ Error!</h4>
                <p style='margin: 5px 0 0 0;'>An error occurred while monitoring the training process: {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
            try:
                (Path('models') / 'training.lock').unlink(missing_ok=True)
            except Exception:
                pass
    
    def load_trained_model(self):
        """Try to load the newly trained model after training completes"""
        try:
            # Short delay to ensure file system has updated
            time.sleep(1)
            
            # Check if the model file exists now
            manager = self.model_manager
            latest = manager.get_latest_model(model_type="cnn") or manager.get_latest_model(None)
            if latest and latest.get('path') and os.path.exists(latest['path']):
                self.model = tf.keras.models.load_model(latest['path'])
                self.model_path = latest['path']
                self.loaded = True
                st.success("✅ Newly trained model loaded successfully!")
                st.rerun()
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
                        st.rerun()
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

    def process_audio_thread(self):
        """Background thread for processing audio in real-time"""
        try:
            while self.real_time_processing:
                # Check for model availability
                if not self.model or not self.loaded:
                    time.sleep(1)
                    continue
                
                # Process any queued audio data
                time.sleep(0.1)  # Prevent excessive CPU usage
                
        except Exception as e:
            logger.error(f"Error in audio processing thread: {e}")
            st.error("Audio processing thread encountered an error and stopped.")

    def create_interactive_visualization(self, confidence_scores):
        """Create an interactive bar chart for emotion confidence scores"""
        fig = go.Figure()

        # Handle empty or missing scores gracefully
        if not confidence_scores:
            fig.update_layout(
                title='No prediction available',
                xaxis_title='Confidence (%)',
                yaxis_title='Emotion',
                height=220,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(range=[0, 100]),
                showlegend=False,
                plot_bgcolor='white'
            )
            return fig

        # Sort emotions by confidence
        emotions = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
        labels, values = zip(*emotions)

        # Create bar chart
        fig.add_trace(go.Bar(
            y=list(labels),
            x=list(values),
            orientation='h',
            marker=dict(
                color=[EMOTION_COLORS.get(emotion, '#607D8B') for emotion in labels],
                line=dict(color='white', width=1)
            ),
            hovertemplate='%{x:.1f}% confidence<extra></extra>'
        ))

        # Update layout
        fig.update_layout(
            title='Emotion Confidence Scores',
            xaxis_title='Confidence (%)',
            yaxis_title='Emotion',
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(range=[0, 100]),
            showlegend=False,
            plot_bgcolor='white'
        )

        return fig
    
    def create_gauge_chart(self, confidence, emotion):
        """Create a gauge chart for primary emotion confidence"""
        colors = {
            'low': '#F87171',
            'medium': '#FCD34D',
            'high': '#4ADE80'
        }
        
        # Determine color based on confidence
        if confidence > 80:
            color = colors['high']
        elif confidence > 50:
            color = colors['medium']
        else:
            color = colors['low']
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': '#FEE2E2'},
                    {'range': [50, 80], 'color': '#FEF3C7'},
                    {'range': [80, 100], 'color': '#D1FAE5'}
                ]
            },
            title={'text': f"Confidence in {emotion.capitalize()}", 'font': {'size': 16}},
            number={'suffix': "%", 'font': {'size': 24}}
        ))
        
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig


# Initialize session state defaults
if 'tensorboard_running' not in st.session_state:
    st.session_state['tensorboard_running'] = False

if 'real_time_enabled' not in st.session_state:
    st.session_state['real_time_enabled'] = False


if __name__ == '__main__':
    # Streamlit execution path
    _app = EmotionAnalyzer()
    _app.run()


