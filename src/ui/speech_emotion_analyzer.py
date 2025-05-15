import os
import logging
import numpy as np
import streamlit as st
import tensorflow as tf
from pathlib import Path
import librosa
import plotly.graph_objects as go
import sys

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.model_manager import ModelManager
from src.models.emotion_model import EmotionModel
from src.features.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

class SpeechEmotionAnalyzer:
    """Main application class for the Speech Emotion Recognition System"""
    
    def __init__(self):
        """Initialize the application components"""
        self.model = None
        self.loaded = False
        self.model_type = "cnn"  # Default model type
        self.model_manager = ModelManager()
        self.feature_extractor = FeatureExtractor()
        self.emotion_labels = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Try to load the latest model
        self.initialize()
    
    def initialize(self):
        """Initialize the analyzer and load a model"""
        try:
            # First try to find latest model in registry
            latest_model = self.model_manager.get_latest_model(model_type=self.model_type)
            if latest_model:
                logger.info(f"Found latest model: {latest_model['id']}")
                self.model = self.model_manager.load_model(model_id=latest_model['id'])
                if self.model:
                    self.model_path = latest_model['path']
                    self.model_type = latest_model['type']
                    self.loaded = True
                    logger.info(f"Successfully loaded {self.model_type.upper()} model from {self.model_path}")
                    return True
            
            # If no model found in registry, try scanning model directory
            self.model_manager._scan_for_new_models()
            latest_model = self.model_manager.get_latest_model(model_type=self.model_type)
            if latest_model:
                self.model = self.model_manager.load_model(model_id=latest_model['id'])
                if self.model:
                    self.model_path = latest_model['path']
                    self.loaded = True
                    logger.info(f"Successfully loaded newly found {self.model_type.upper()} model")
                    return True
            
            # If still no model, try scanning logs directory
            latest_model = self._scan_logs_for_model()
            if latest_model:
                return True
                
            # If no model found, create a default one
            if not os.path.exists(os.path.join('models', 'cnn_emotion_model.keras')):
                logger.warning("No trained model found. Creating a default model...")
                emotion_model = EmotionModel(num_classes=len(self.emotion_labels))
                cnn_model = emotion_model.build_cnn(input_shape=(128, 165, 1))
                cnn_model.save('models/cnn_emotion_model.keras')
                logger.info("Created default model. Please train it before using.")
                
            st.warning("⚠️ No trained model found. Please train a model using main.py first.")
            return False
            
        except Exception as e:
            logger.error(f"Error initializing analyzer: {e}")
            st.error(f"❌ Error loading model: {str(e)}")
            return False
    
    def _scan_and_load_model(self):
        """Scan models directory and load the latest model"""
        try:
            # Scan for new models first
            self.model_manager._scan_for_new_models()
            
            # Get latest model
            latest_model = self.model_manager.get_latest_model(model_type=self.model_type)
            if latest_model:
                logger.info(f"Found latest model: {latest_model['id']}")
                self.model = self.model_manager.load_model(model_id=latest_model['id'])
                if self.model:
                    self.model_path = latest_model['path']
                    self.loaded = True
                    logger.info(f"Successfully loaded {self.model_type.upper()} model from {self.model_path}")
                    return True
            return False
            
        except Exception as e:
            logger.error(f"Error scanning for models: {e}")
            return False
            
    def _scan_logs_for_model(self):
        """Scan logs directory for trained models"""
        try:
            log_dir = Path('logs')
            if not log_dir.exists():
                return False
                
            # Find most recent run directory
            run_dirs = sorted(log_dir.glob('run_*'), reverse=True)
            for run_dir in run_dirs:
                model_path = run_dir / 'best_model.keras'
                if model_path.exists():
                    try:
                        self.model = tf.keras.models.load_model(str(model_path))
                        self.model_path = str(model_path)
                        self.loaded = True
                        logger.info(f"Loaded model from {model_path}")
                        
                        # Register model in the manager
                        self.model_manager.register_model(
                            model_path=str(model_path),
                            model_type=self.model_type,
                            description=f"Model loaded from {run_dir.name}"
                        )
                        return True
                    except Exception as e:
                        logger.warning(f"Failed to load model from {model_path}: {e}")
            return False
            
        except Exception as e:
            logger.error(f"Error scanning logs directory: {e}")
            return False
            
    def extract_features(self, audio_file):
        """Extract features from audio file for model prediction"""
        try:
            # Load and preprocess audio
            y, sr = librosa.load(audio_file)
            
            # Display waveform
            self.visualize_waveform(y, sr)
            
            # Extract features
            features = self.feature_extractor.extract_features(y, sr)
            if features is None:
                st.error("❌ Failed to extract features from audio!")
                return None, None, None
                
            return y, sr, features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            st.error(f"❌ Error processing audio: {str(e)}")
            return None, None, None
            
    def predict_emotion(self, features):
        """Predict emotion from audio features"""
        try:
            if not self.loaded or self.model is None:
                st.error("❌ Model is not loaded! Cannot make predictions.")
                return None, None
                
            predictions = self.model.predict(features, verbose=0)
            emotion_probs = dict(zip(self.emotion_labels, predictions[0]))
            predicted_emotion = max(emotion_probs.items(), key=lambda x: x[1])[0]
            
            return predicted_emotion, emotion_probs
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            st.error(f"❌ Error predicting emotion: {str(e)}")
            return None, None
            
    def visualize_waveform(self, y, sr):
        """Create and display audio waveform visualization"""
        try:
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
            
        except Exception as e:
            logger.error(f"Error visualizing waveform: {e}")
            st.error(f"❌ Error creating waveform visualization: {str(e)}")
            
    def display_emotion_probabilities(self, emotion_probs):
        """Display emotion prediction probabilities"""
        try:
            st.subheader("🎯 Emotion Predictions")
            
            # Sort probabilities for better visualization
            sorted_probs = dict(sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True))
            
            # Create bars for each emotion
            for emotion, prob in sorted_probs.items():
                prob_percentage = prob * 100
                st.write(f"{emotion.capitalize()}")
                st.progress(prob)
                st.write(f"{prob_percentage:.2f}%")
                
        except Exception as e:
            logger.error(f"Error displaying probabilities: {e}")
            st.error(f"❌ Error displaying results: {str(e)}")
            
    def process_audio(self, audio_file):
        """Process audio file and predict emotion"""
        try:
            if not self.loaded or self.model is None:
                st.warning("⚠️ No trained model found. Please train a model first using:")
                st.code("python -m src.main --train --model-type cnn")
                return
                
            # Extract features
            y, sr, features = self.extract_features(audio_file)
            if y is None or features is None:
                return
                
            # Make prediction
            emotion, probs = self.predict_emotion(features)
            if emotion is None:
                return
                
            # Display results
            self.display_emotion_probabilities(probs)
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            st.error(f"❌ Error processing audio: {str(e)}")

    def run(self):
        """Run the Streamlit application"""
        st.title("🎭 Speech Emotion Recognition System")
        st.write("Upload an audio file or use the demo samples to analyze emotions in speech.")
        
        # File uploader
        audio_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3'])
        
        # Demo section
        st.subheader("🎯 Try a Demo")
        demo_dir = Path("demo_files")
        if demo_dir.exists():
            demo_files = list(demo_dir.glob("*.wav"))
            cols = st.columns(len(demo_files))
            
            for idx, demo_file in enumerate(demo_files):
                emotion = demo_file.stem.split('_')[0]
                with cols[idx]:
                    if st.button(f"Test {emotion.capitalize()} Audio"):
                        st.audio(str(demo_file))
                        self.process_audio(str(demo_file))
        
        # Process uploaded file
        if audio_file:
            st.audio(audio_file)
            self.process_audio(audio_file)

def main():
    """Main entry point for the application"""
    app = SpeechEmotionAnalyzer()
    app.run()

if __name__ == "__main__":
    main()
