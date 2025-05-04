import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa
import librosa.display
import soundfile as sf
import time
from pathlib import Path
from streamlit_option_menu import option_menu
from streamlit_extras.colored_header import colored_header
from streamlit_extras.app_logo import add_logo
from streamlit_extras.card import card
from streamlit_extras.stylable_container import stylable_container
import plotly.express as px

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
        self.ensure_upload_dir()
        
    def ensure_upload_dir(self):
        """Ensure the upload directory exists"""
        os.makedirs(self.upload_folder, exist_ok=True)
    
    def load_model(self):
        """Load the pre-trained emotion model"""
        try:
            with st.spinner("Loading model... Please wait."):
                try:
                    self.model = tf.keras.models.load_model(self.model_path)
                    st.success("Model loaded successfully!")
                except Exception as e:
                    st.warning(f"Failed to load model from {self.model_path}. Trying backup model...")
                    try:
                        self.model = tf.keras.models.load_model(self.backup_model_path)
                        st.success("Backup model loaded successfully!")
                    except Exception as e2:
                        st.error(f"Failed to load backup model: {e2}")
                        st.stop()
            
            self.loaded = True
        except Exception as e:
            st.error(f"Error loading model: {e}")
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
                
                # Extract mel spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128, fmax=8000
                )
                
                # Convert to decibels
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                # Reshape for model input (adding batch and channel dimensions)
                mel_spec_db = mel_spec_db.reshape(1, mel_spec_db.shape[0], mel_spec_db.shape[1], 1)
                
                return y, sr, mel_spec_db
        except Exception as e:
            st.error(f"Error extracting features: {e}")
            return None, None, None
    
    def predict_emotion(self, features):
        """Predict emotion from audio features"""
        if not self.loaded:
            self.load_model()
            
        try:
            with st.spinner("Predicting emotion..."):
                # Make prediction
                prediction = self.model.predict(features)
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
    
    def display_file_upload(self):
        """Display file upload interface and handle uploaded files"""
        with st.container():
            colored_header(
                label="Upload Audio",
                description="Upload an audio file to analyze the speaker's emotions",
                color_name="violet-70"
            )
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                with stylable_container(
                    key="upload_container",
                    css_styles="""
                        {
                            background-color: #f0f2f6;
                            border-radius: 10px;
                            padding: 20px;
                        }
                    """
                ):
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
            
            with col2:
                with stylable_container(
                    key="record_container",
                    css_styles="""
                        {
                            background-color: #ede7f6;
                            border-radius: 10px;
                            padding: 20px;
                        }
                    """
                ):
                    st.write("### Record Audio")
                    st.write("Record your voice directly in the browser:")
                    
                    audio_recording = st.audio_recorder(
                        text="Click to record",
                        recording_color="#e65100",
                        neutral_color="#5e35b1",
                        sample_rate=16000,
                    )
            
            # Process the uploaded file or recorded audio
            if uploaded_file is not None:
                # Save the uploaded file to disk
                file_path = os.path.join(self.upload_folder, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.success(f"File uploaded successfully: {uploaded_file.name}")
                self.process_audio(file_path)
                
            elif audio_recording is not None:
                # Save the recorded audio to disk
                timestamp = int(time.time())
                file_path = os.path.join(self.upload_folder, f"recording_{timestamp}.wav")
                with open(file_path, "wb") as f:
                    f.write(audio_recording)
                
                st.success("Recording saved successfully!")
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
        
        # Extract features from audio
        y, sr, features = self.extract_features(audio_file_path)
        
        if features is None:
            st.error("Failed to extract features from the audio file.")
            return
        
        # Predict emotion
        emotion, confidence_scores = self.predict_emotion(features)
        
        # Display results
        self.display_results(audio_file_path, y, sr, emotion, confidence_scores)
    
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
                    
                    # Display most confident emotion with progress bar
                    max_confidence = max(confidence_scores.values())
                    st.markdown(f"Confidence: **{max_confidence:.1f}%**")
                    st.progress(max_confidence/100)
                    
                    # Display emotion description
                    st.markdown(f"**Description**: {EMOTION_DESCRIPTIONS.get(emotion, '')}")
                    
                    # Display original audio for playback
                    st.audio(audio_file_path)
            
            with col2:
                tabs = st.tabs(["Confidence Scores", "Audio Visualization"])
                
                with tabs[0]:
                    # Display interactive bar chart of confidence scores
                    fig = self.create_interactive_visualization(confidence_scores)
                    st.plotly_chart(fig, use_container_width=True)
                
                with tabs[1]:
                    # Display audio visualizations
                    fig = self.visualize_audio(y, sr)
                    st.pyplot(fig)
    
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
    
    def run(self):
        """Main method to run the Streamlit application"""
        # Display app header
        st.markdown("<h1 class='main-header'>Speech Emotion Analyzer</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 1.2rem;'>Detect emotions in speech using AI</p>", unsafe_allow_html=True)
        
        # Add logo
        # add_logo("assets/logo.png")
        
        # Sidebar navigation
        with st.sidebar:
            st.image("results/visualizations/enhanced_confusion_matrix.png", caption="Emotion Classification Matrix", use_column_width=True)
            
            selected = option_menu(
                menu_title="Navigation",
                options=["Analyze Audio", "View Examples", "About", "Settings"],
                icons=["soundwave", "play-circle", "info-circle", "gear"],
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
        
        # Display selected section
        if selected == "Analyze Audio":
            self.display_file_upload()
            
        elif selected == "View Examples":
            self.display_demo_section()
            
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
                if st.button("Save Settings"):
                    self.model_path = model_path
                    st.success("Settings saved successfully!")
                    self.loaded = False  # Force model reload with new settings
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<p style='text-align: center; color: gray;'>Speech Emotion Analyzer v1.0 | Made with ❤️ using Streamlit</p>",
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    app = EmotionAnalyzer()
    app.run()