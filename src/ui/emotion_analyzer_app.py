import os
import logging
import numpy as np
import streamlit as st
import tensorflow as tf
from pathlib import Path
import librosa
import plotly.graph_objects as go

from src.models.model_manager import ModelManager 
from src.features.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

class EmotionAnalyzerApp:
    """Main application class for the Speech Emotion Analyzer"""
    
    def __init__(self):
        """Initialize the application components"""
        self.model = None
        self.loaded = False
        self.model_type = "cnn"  # Default model type
        self.model_manager = ModelManager()
        self.feature_extractor = FeatureExtractor()
        self.emotion_labels = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Try to load the latest model
        self.load_latest_model()
    
    def load_latest_model(self):
        """Load the most recent model from the registry"""
        try:
            latest_model = self.model_manager.get_latest_model(model_type="cnn")
            if latest_model:
                logger.info(f"Found latest model: {latest_model['id']}")
                self.model = self.model_manager.load_model(model_id=latest_model['id'])
                if self.model:
                    self.model_path = latest_model['path']
                    self.model_type = latest_model['type']
                    self.loaded = True
                    logger.info(f"Successfully loaded {self.model_type.upper()} model from {self.model_path}")
                    return True
            
            # If no model in registry, try scanning model directory
            self.model_manager._scan_for_new_models()
            latest_model = self.model_manager.get_latest_model(model_type="cnn")
            if latest_model:
                self.model = self.model_manager.load_model(model_id=latest_model['id'])
                if self.model:
                    self.model_path = latest_model['path']
                    self.model_type = latest_model['type']
                    self.loaded = True
                    logger.info(f"Successfully loaded newly found {self.model_type.upper()} model")
                    return True
            
            logger.warning("No models found in registry or models directory")
            return False
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def extract_features(self, audio_file):
        """Extract features from audio file for model prediction"""
        try:
            # Load and preprocess audio
            y, sr = librosa.load(audio_file)
            
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
            if not self.loaded:
                st.error("❌ Model is not loaded! Cannot make predictions.")
                return None
            
            predictions = self.model.predict(features, verbose=0)
            emotion_probabilities = dict(zip(self.emotion_labels, predictions[0]))
            predicted_emotion = max(emotion_probabilities.items(), key=lambda x: x[1])[0]
            
            return predicted_emotion, emotion_probabilities
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            st.error(f"❌ Error predicting emotion: {str(e)}")
            return None

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
            if not self.loaded:
                st.warning("⚠️ No model loaded. Please train a model first.")
                return

            # Extract features and visualize audio
            y, sr, features = self.extract_features(audio_file)
            if y is None or features is None:
                return
            
            # Display waveform
            self.visualize_waveform(y, sr)
            
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
        st.title("🎭 Speech Emotion Analyzer")
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
    app = EmotionAnalyzerApp()
    app.run()

if __name__ == "__main__":
    main()
