#!/usr/bin/env python3
# fixed_app.py - Fixed version of the Speech Emotion Analyzer App

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import streamlit as st
from streamlit_option_menu import option_menu
import librosa
import librosa.display
import tensorflow as tf
import time
import plotly.express as px

# Configure logging (ensure it's defined or imported if used elsewhere, e.g. main.py)
# For this file, if logging is specific to its execution as a script:
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("fixed_app.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
else: # If imported as a module
    logger = logging.getLogger(__name__)


# Model path (ensure this is correct and accessible)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "emotion_model.h5")
MODEL_INFO_PATH = os.path.join(os.path.dirname(__file__), "models", "model_info.json")


class EmotionAnalyzer:
    def __init__(self, model_path=MODEL_PATH, model_info_path=MODEL_INFO_PATH):
        self.model_path = model_path
        self.model_info_path = model_info_path
        self.model = self._load_model()
        self.model_info = self._load_model_info()
        # Define emotion labels based on your model's output
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'] # Example

    def _load_model(self):
        model_dir = os.path.dirname(self.model_path)
        preferred_model_path = self.model_path
        loaded_model = None
        model_loaded_successfully = False

        if os.path.exists(preferred_model_path):
            logger.info(f"Attempting to load preferred model: {preferred_model_path}")
            try:
                loaded_model = tf.keras.models.load_model(preferred_model_path)
                logger.info(f"Preferred model loaded successfully from {preferred_model_path}")
                if 'st' in sys.modules: # Check if streamlit is imported before using st functions
                    st.info(f"Using preferred model: {os.path.basename(preferred_model_path)}")
                model_loaded_successfully = True
            except Exception as e:
                logger.error(f"Error loading preferred model {preferred_model_path}: {e}. Will try to find other models.")
                if 'st' in sys.modules:
                    st.warning(f"Could not load preferred model {os.path.basename(preferred_model_path)}. Trying other available models.")
        else:
            logger.warning(f"Preferred model file not found at {preferred_model_path}. Attempting to find other models.")

        if not model_loaded_successfully:
            potential_models = []
            if os.path.exists(model_dir):
                for fname in os.listdir(model_dir):
                    if fname.endswith(".keras") or fname.endswith(".h5"):
                        current_file_path = os.path.join(model_dir, fname)
                        # Avoid re-trying the preferred path if it was already attempted
                        if current_file_path == preferred_model_path and os.path.exists(preferred_model_path):
                            continue
                        potential_models.append(current_file_path)
            
            if not potential_models:
                logger.error(f"No other model files (.h5 or .keras) found in {model_dir} to try.")
                if not os.path.exists(preferred_model_path) and 'st' in sys.modules:
                     st.error(f"No model files found in the '{os.path.basename(model_dir)}' directory.")
                # If preferred existed but failed, the st.warning about it is already shown.
                # A final error will be shown if all attempts fail.
            else:
                # Sort: .keras first, then by name
                sorted_models = sorted(potential_models, key=lambda x: (not x.endswith(".keras"), x))
                
                for model_file_to_try in sorted_models:
                    logger.info(f"Attempting to load alternative model: {model_file_to_try}")
                    try:
                        loaded_model = tf.keras.models.load_model(model_file_to_try)
                        logger.info(f"Alternative model loaded successfully from {model_file_to_try}")
                        if 'st' in sys.modules:
                            st.success(f"Successfully loaded alternative model: {os.path.basename(model_file_to_try)}")
                        model_loaded_successfully = True
                        break # Stop after loading one successfully
                    except Exception as e:
                        logger.error(f"Error loading alternative model {model_file_to_try}: {e}")
                        # Continue to try the next model
            
            if not model_loaded_successfully and 'st' in sys.modules:
                st.error(f"Failed to load any model from the '{os.path.basename(model_dir)}' directory.")
                return None # Return None if no model could be loaded
        
        return loaded_model

    def _load_model_info(self):
        if os.path.exists(self.model_info_path):
            try:
                with open(self.model_info_path, 'r') as f:
                    info = pd.read_json(f, typ='series').to_dict() # More robust loading
                logger.info(f"Model info loaded successfully from {self.model_info_path}")
                return info
            except Exception as e:
                logger.error(f"Error loading model info: {e}")
                st.warning(f"Could not load model info: {e}")
                return {} # Return empty dict on error
        else:
            logger.warning(f"Model info file not found at {self.model_info_path}")
            return {} # Return empty dict if not found

    def predict_emotion(self, audio_path):
        if self.model is None:
            st.error("Model is not loaded. Cannot predict.")
            return "Error", {"Error": 1.0}
        try:
            data, sample_rate = librosa.load(audio_path, duration=3, offset=0.5, sr=44100)
            
            # Pad or truncate to a fixed length if your model expects it
            # Example: fixed length of 3 seconds at 44100 Hz = 132300 samples
            expected_audio_length = 132300 
            if len(data) < expected_audio_length:
                data = np.pad(data, (0, expected_audio_length - len(data)), 'constant')
            elif len(data) > expected_audio_length:
                data = data[:expected_audio_length]

            # Revised MFCC processing to match model's expected input shape (None, 128, 165, 1)
            n_mfcc_expected = 128
            n_frames_expected = 165

            # Extract MFCCs. Default hop_length=512. For 3s audio at 44.1kHz (132300 samples),
            # this yields approx. 132300/512 = 258 frames.
            raw_mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=n_mfcc_expected) # Shape: (n_mfcc_expected, actual_frames)

            # Pad or truncate the time frames dimension (axis 1) to n_frames_expected
            current_frames = raw_mfccs.shape[1]
            if current_frames > n_frames_expected:
                mfccs_processed = raw_mfccs[:, :n_frames_expected]
            elif current_frames < n_frames_expected:
                padding_width = n_frames_expected - current_frames
                mfccs_processed = np.pad(raw_mfccs, ((0, 0), (0, padding_width)), mode='constant')
            else:
                mfccs_processed = raw_mfccs
            # mfccs_processed shape is now (n_mfcc_expected, n_frames_expected) -> (128, 165)

            # Expand dimensions for the model: (batch_size, height, width, channels)
            mfccs_final = np.expand_dims(mfccs_processed, axis=0) # Shape: (1, 128, 165)
            mfccs_final = np.expand_dims(mfccs_final, axis=-1)    # Shape: (1, 128, 165, 1)

            prediction = self.model.predict(mfccs_final)
            predicted_emotion_index = np.argmax(prediction)
            
            if predicted_emotion_index < len(self.emotion_labels):
                predicted_emotion = self.emotion_labels[predicted_emotion_index]
            else:
                st.error(f"Predicted index {predicted_emotion_index} is out of bounds for emotion labels.")
                return "Error", {"Error": 1.0}

            probabilities = {self.emotion_labels[i]: float(prediction[0][i]) for i in range(len(self.emotion_labels))}
            return predicted_emotion, probabilities

        except Exception as e:
            logger.error(f"Error during emotion prediction: {e}")
            st.error(f"Error during emotion prediction: {e}")
            return "Error", {"Error": 1.0}

    def plot_emotion_probabilities(self, probabilities, predicted_emotion):
        if not probabilities or "Error" in probabilities:
            st.warning("Cannot plot probabilities due to prediction error or no data.")
            return

        try:
            labels = list(probabilities.keys())
            probs = list(probabilities.values())

            fig, ax = plt.subplots(figsize=(10, 6))
            # Use a color palette that works well in dark mode
            bar_colors = ['#1f77b4' if label != predicted_emotion else '#ff7f0e' for label in labels] # Blue and Orange
            bars = ax.bar(labels, probs, color=bar_colors)
            
            ax.set_xlabel("Emotions", fontsize=12, color='white')
            ax.set_ylabel("Probability", fontsize=12, color='white')
            ax.set_title("Emotion Probabilities", fontsize=15, color='white')
            ax.tick_params(axis='x', rotation=45, colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.set_facecolor('#0E1117') # Match streamlit dark background
            fig.patch.set_facecolor('#0E1117')


            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', color='white')
            
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            logger.error(f"Error plotting emotion probabilities: {e}")
            st.error(f"Error plotting probabilities: {e}")

    def plot_waveform_and_spectrogram(self, audio_path):
        try:
            y, sr = librosa.load(audio_path)
            
            fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(10, 8))
            fig.patch.set_facecolor('#0E1117') # Match streamlit dark background

            # Waveform
            librosa.display.waveshow(y, sr=sr, ax=axes[0], color="cyan")
            axes[0].set_title('Waveform', color='white')
            axes[0].set_ylabel('Amplitude', color='white')
            axes[0].tick_params(colors='white')
            axes[0].set_facecolor('#0E1117')


            # Spectrogram
            D = librosa.stft(y)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=axes[1], cmap='magma')
            axes[1].set_title('Spectrogram (dB)', color='white')
            axes[1].set_ylabel('Frequency (Hz)', color='white')
            axes[1].set_xlabel('Time (s)', color='white')
            axes[1].tick_params(colors='white')
            axes[1].set_facecolor('#0E1117')
            
            cb = fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
            cb.ax.yaxis.set_tick_params(color='white')
            cb.outline.set_edgecolor('white')
            plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')


            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            logger.error(f"Error plotting waveform/spectrogram: {e}")
            st.error(f"Error plotting waveform/spectrogram: {e}")

    def process_audio(self, audio_path):
        st.subheader("Audio Analysis")
        predicted_emotion, probabilities = self.predict_emotion(audio_path)

        if predicted_emotion != "Error":
            st.markdown(f"**Predicted Emotion:** <span style='color: #ff7f0e; font-size: 1.2em;'>{predicted_emotion.capitalize()}</span>", unsafe_allow_html=True)
        else:
            st.error("Could not predict emotion from the audio.")
            return

        st.audio(audio_path, format="audio/wav")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Waveform & Spectrogram")
            self.plot_waveform_and_spectrogram(audio_path)
        
        with col2:
            st.markdown("##### Emotion Probabilities")
            if probabilities and predicted_emotion != "Error":
                 self.plot_emotion_probabilities(probabilities, predicted_emotion)
            else:
                st.info("Probability plot is unavailable.")


class StreamlitApp:
    def __init__(self, model_path_arg=MODEL_PATH, model_info_path_arg=MODEL_INFO_PATH):
        self.model_path_arg = model_path_arg
        self.model_info_path_arg = model_info_path_arg
        self.analyzer = None # Will be initialized in run()

    def run(self):
        st.set_page_config(layout="wide", page_title="Speech Emotion Analyzer", page_icon="🗣️")

        if "nav_selection" not in st.session_state:
            st.session_state.nav_selection = "Home"
        
        # Initialize the analyzer here, after set_page_config
        if self.analyzer is None:
            self.analyzer = EmotionAnalyzer(self.model_path_arg, self.model_info_path_arg)

        # Apply custom CSS
        st.markdown(
            """
            <style>
                /* General body style */
                body {
                    color: #E0E0E0; 
                    background-color: #0E1117; 
                }
                /* Sidebar styling */
                [data-testid="stSidebar"] > div:first-child {
                    background-color: #1A1D21; 
                    color: #E0E0E0; 
                }
                [data-testid="stSidebar"] .st-emotion-cache-16idsys a, 
                [data-testid="stSidebar"] .st-emotion-cache-16idsys span { /* Target both links and spans for icons */
                    color: #E0E0E0 !important;
                }
                [data-testid="stSidebar"] .st-emotion-cache-16idsys a:hover,
                [data-testid="stSidebar"] .st-emotion-cache-16idsys span:hover {
                    color: #FFFFFF !important;
                    background-color: #2C3038; /* Hover background for nav items */
                }
                 /* Active nav item */
                [data-testid="stSidebar"] .st-emotion-cache-16idsys li[data-active="true"] a {
                    color: #4CAF50 !important; /* Accent color for active item */
                    font-weight: bold;
                }


                /* Model info items in sidebar */
                .model-info-container {
                    padding: 10px;
                }
                .model-info-item {
                    background-color: #2C3038; 
                    color: #E0E0E0; 
                    border-radius: 8px;
                    padding: 12px;
                    margin-bottom: 10px;
                    border: 1px solid #3A3F4A; 
                }
                .model-info-item strong {
                    color: #FFFFFF; 
                    font-weight: bold;
                }

                /* Action card styling */
                .action-card-container {
                    display: flex;
                    justify-content: space-around;
                    gap: 20px; 
                    padding: 20px;
                    flex-wrap: wrap; 
                }
                .action-card {
                    background-color: #1A1D21; 
                    border-radius: 10px;
                    padding: 25px;
                    text-align: center;
                    width: 300px; 
                    min-height: 280px; /* Ensure cards have same height */
                    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
                    border: 1px solid #3A3F4A;
                    color: #E0E0E0; 
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between; /* Pushes button to bottom */
                }
                .action-card h3 {
                    color: #FFFFFF; 
                    margin-bottom: 15px;
                    font-size: 1.5em;
                }
                .action-card p {
                    color: #B0B0B0; 
                    font-size: 0.95em;
                    margin-bottom: 20px;
                    flex-grow: 1; /* Allows paragraph to take available space */
                }
                .card-icon {
                    font-size: 3.5em; 
                    margin-bottom: 20px;
                    color: #4CAF50; 
                }
                .action-card .stButton button {
                    background-color: #4CAF50;
                    color: white;
                    border-radius: 5px;
                    padding: 12px 24px;
                    border: none;
                    font-weight: bold;
                    width: 100%; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                    transition: background-color 0.3s ease;
                }
                .action-card .stButton button:hover {
                    background-color: #45a049;
                }

                /* General Streamlit element styling for dark theme */
                h1, h2, h3, h4, h5, h6 { color: #FFFFFF; }
                p, div, span, li, label, .stMarkdown, .stAlert, .stMetric, .stJson, .stDataFrame, .stTable {
                     color: #E0E0E0 !important;
                }
                .stTextInput > div > div > input, .stFileUploader > div > div > button {
                    color: #E0E0E0;
                    background-color: #2C3038;
                }
                /* Ensure text in notifications is visible */
                 div[data-testid="stNotificationContentError"],
                 div[data-testid="stNotificationContentWarning"],
                 div[data-testid="stNotificationContentInfo"],
                 div[data-testid="stNotificationContentSuccess"] {
                    color: #1E1E1E !important; /* Dark text for default light notification boxes */
                 }
                 /* Custom styled success/error messages if needed */
                .stAlert[data-baseweb="alert"] > div:first-child { /* More specific selector for alert text */
                    color: inherit !important; /* Inherit from .stAlert if set, or use default */
                }


            </style>
            """,
            unsafe_allow_html=True,
        )

        with st.sidebar:
            st.markdown("<h1 style='color: #FFFFFF; text-align: center;'>SER</h1>", unsafe_allow_html=True)
            st.markdown("<p style='color: #B0B0B0; text-align: center; margin-bottom: 20px;'>Speech Emotion Recognition</p>")
            
            # Navigation
            selected = option_menu(
                menu_title=None,  # Required
                options=["Home", "Analyze Audio", "Dashboard", "About"],
                icons=["house-fill", "soundwave", "bar-chart-line-fill", "info-circle-fill"],
                menu_icon="cast", default_index=["Home", "Analyze Audio", "Dashboard", "About"].index(st.session_state.nav_selection),
                styles={
                    "container": {"padding": "5px !important", "background-color": "#1A1D21"},
                    "icon": {"color": "#E0E0E0", "font-size": "20px"},
                    "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#2C3038", "color": "#E0E0E0"},
                    "nav-link-selected": {"background-color": "#0E1117", "color": "#4CAF50 !important", "font-weight": "bold"},
                }
            )
            if selected != st.session_state.nav_selection:
                st.session_state.nav_selection = selected
                st.rerun()


            st.markdown("---")
            st.markdown("<h3 style='color: #FFFFFF;'>Model Information</h3>", unsafe_allow_html=True)
            if self.analyzer.model_info:
                st.markdown(f"<div class='model-info-item'><strong>Name:</strong> {self.analyzer.model_info.get('name', 'N/A')}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='model-info-item'><strong>Version:</strong> {self.analyzer.model_info.get('version', 'N/A')}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='model-info-item'><strong>Accuracy:</strong> {self.analyzer.model_info.get('accuracy', 'N/A')*100:.2f}%</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='model-info-item'><strong>Trained:</strong> {self.analyzer.model_info.get('last_trained', 'N/A')}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='model-info-item'>Model details not available.</div>", unsafe_allow_html=True)

        # Page routing
        if st.session_state.nav_selection == "Home":
            self.display_home_page()
        elif st.session_state.nav_selection == "Analyze Audio":
            self.display_file_upload()
        elif st.session_state.nav_selection == "Dashboard":
            self.display_dashboard_page()
        elif st.session_state.nav_selection == "About":
            self.display_about_page()

    def display_home_page(self):
        st.markdown("<h1 style='text-align: center; color: #FFFFFF;'>Welcome to the Speech Emotion Analyzer</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #B0B0B0; margin-bottom: 30px;'>Navigate using the sidebar or the quick action cards below.</p>", unsafe_allow_html=True)

        # Action cards in a container for better layout control
        st.markdown("<div class='action-card-container'>", unsafe_allow_html=True)
        
        # Card 1: Analyze Audio File
        st.markdown("""
            <div class="action-card">
                <div class="card-icon">🗣️</div>
                <h3>Analyze Audio File</h3>
                <p>Upload an audio file (WAV, MP3) and let the model predict the emotion conveyed in the speech.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Analyze Audio", key="home_analyze", help="Upload and analyze an audio file"):
            st.session_state.nav_selection = "Analyze Audio"
            st.rerun() # Corrected

        # Card 3: View Dashboard
        st.markdown("""
            <div class="action-card">
                <div class="card-icon">📊</div>
                <h3>View Dashboard</h3>
                <p>Explore model performance metrics, visualizations, and other relevant statistics about the analysis.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Dashboard", key="home_dashboard", help="View model performance and data"):
            st.session_state.nav_selection = "Dashboard"
            st.rerun() # Corrected
            
        st.markdown("</div>", unsafe_allow_html=True) # Close action-card-container

    def display_file_upload(self):
        st.header("Analyze Audio from File")
        uploaded_file = st.file_uploader("Upload an audio file (WAV, MP3)", type=["wav", "mp3"], key="file_uploader")
        
        sample_audio_dir = os.path.join(os.path.dirname(__file__), "samples")
        sample_audio_path = os.path.join(sample_audio_dir, "sample1.wav") # Default sample

        if uploaded_file is not None:
            with st.spinner("Processing uploaded audio..."):
                temp_audio_path = os.path.join("/tmp", uploaded_file.name) # Use /tmp for temporary files
                with open(temp_audio_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                self.analyzer.process_audio(temp_audio_path)
                
                if os.path.exists(temp_audio_path):
                    try:
                        os.remove(temp_audio_path)
                    except Exception as e:
                        logger.warning(f"Could not remove temp file {temp_audio_path}: {e}")
        
        st.markdown("---")
        st.markdown("#### Or try with a sample audio:")
        
        if not os.path.exists(sample_audio_path):
            st.warning(f"Sample audio file not found at: {sample_audio_path}. Please check the 'samples' directory.")
        elif st.button("Try a Sample Audio", key="try_sample"):
            with st.spinner("Processing sample audio..."):
                self.analyzer.process_audio(sample_audio_path)

    def display_dashboard_page(self):
        st.header("Application Dashboard")
        st.markdown("This section provides an overview of model performance and data insights.")

        # Placeholder for dashboard content
        # Example: Displaying model accuracy from model_info
        if self.analyzer.model_info and 'accuracy' in self.analyzer.model_info:
            accuracy = self.analyzer.model_info['accuracy'] * 100
            st.metric(label="Model Accuracy", value=f"{accuracy:.2f}%")
        else:
            st.info("Model accuracy information is not available.")

        # Example: Dummy data for a chart
        st.subheader("Emotion Distribution (Sample Data)")
        try:
            # Create some sample data for demonstration
            data = {
                'Emotion': self.analyzer.emotion_labels,
                'Count': np.random.randint(10, 100, size=len(self.analyzer.emotion_labels))
            }
            df = pd.DataFrame(data)
            
            fig = px.bar(df, x='Emotion', y='Count', title="Sample Emotion Distribution", color='Emotion')
            fig.update_layout(plot_bgcolor='#0E1117', paper_bgcolor='#0E1117', font_color='white')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not display sample chart: {e}")
            logger.error(f"Dashboard chart error: {e}")


        st.markdown("---")
        st.subheader("Further Analysis (Placeholder)")
        st.info("More detailed visualizations and statistics will be added here, such as confusion matrices, precision/recall curves, or analysis of prediction confidence over time.")


    def display_about_page(self):
        st.header("About This Application")
        st.markdown("""
            This Speech Emotion Recognition (SER) application is designed to analyze audio inputs 
            (either from uploaded files or live recordings) and predict the underlying emotion.

            **Features:**
            - **Audio File Analysis:** Upload WAV or MP3 files for emotion prediction.
            - **Visualizations:** View waveforms, spectrograms, and emotion probability distributions.
            - **Dashboard:** (Under development) Intended to show model performance metrics.

            **Model:**
            The application uses a deep learning model trained to classify emotions from speech. 
            Details about the model architecture, dataset, and performance can be found in the 
            project's documentation or the 'Model Information' section in the sidebar.

            **Technology Stack:**
            - **Streamlit:** For the web application interface.
            - **Librosa:** For audio processing and feature extraction.
            - **TensorFlow/Keras:** For the deep learning model.
            - **Matplotlib & Plotly:** For generating visualizations.

            **Developed by:** GitHub Copilot & You!
            
            For more information, please refer to the project's README file or contact the developers.
        """)
        st.markdown("---")
        st.markdown("Version: 1.1.0 (UI Overhaul)")


if __name__ == "__main__":
    # Ensure 'models' and 'samples' directories exist relative to fixed_app.py
    base_dir = os.path.dirname(__file__)
    models_dir = os.path.join(base_dir, "models")
    samples_dir = os.path.join(base_dir, "samples")
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logger.info(f"Created directory: {models_dir}")
        
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
        logger.info(f"Created directory: {samples_dir}")

    # Create a dummy model_info.json if it doesn't exist for basic app functionality
    if not os.path.exists(MODEL_INFO_PATH):
        logger.warning(f"{MODEL_INFO_PATH} not found. Creating a dummy file.")
        dummy_info = {
            "name": "DefaultCNN", "version": "0.1.0", 
            "accuracy": 0.0, "last_trained": "N/A",
            "features": "MFCC", "classes": ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        }
        try:
            with open(MODEL_INFO_PATH, 'w') as f:
                import json
                json.dump(dummy_info, f, indent=4)
        except Exception as e:
            logger.error(f"Could not create dummy model_info.json: {e}")

    # Create a dummy sample1.wav if it doesn't exist
    sample_audio_path = os.path.join(samples_dir, "sample1.wav")
    if not os.path.exists(sample_audio_path):
        logger.warning(f"{sample_audio_path} not found. Creating a dummy silent WAV file.")
        try:
            # Create a 1-second silent WAV file
            sr_sample = 22050
            duration_sample = 1
            silence = np.zeros(int(sr_sample * duration_sample))
            import soundfile as sf # Requires soundfile library: pip install soundfile
            sf.write(sample_audio_path, silence, sr_sample)
            logger.info(f"Created dummy sample audio: {sample_audio_path}")
        except ImportError:
            logger.error("Python package 'soundfile' is not installed. Cannot create dummy WAV. Please install it: pip install soundfile")
        except Exception as e:
            logger.error(f"Could not create dummy sample1.wav: {e}")


    app = StreamlitApp()
    app.run()
