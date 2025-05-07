import os
import numpy as np
import shutil
import urllib.request
import zipfile
import tensorflow as tf
from tqdm import tqdm

def download_sample_files():
    """Download sample audio files for demo purposes"""
    print("Setting up demo files for the Speech Emotion Classification app...")
    
    # Create demo directory
    os.makedirs("demo_files", exist_ok=True)
    
    # Check if we already have the demo files
    if os.path.exists("demo_files/happy_sample.wav") and \
       os.path.exists("demo_files/angry_sample.wav") and \
       os.path.exists("demo_files/sad_sample.wav"):
        print("Demo files already exist. Skipping download.")
        return
    
    # URLs for sample audio files (these are placeholders - replace with your actual URLs)
    # For this demo, we'll use sample files from the RAVDESS dataset
    sample_files = {
        "happy_sample.wav": "https://zenodo.org/records/8381238/files/03-01-08-01-01-02-01.wav",
        "angry_sample.wav": "https://zenodo.org/records/8381238/files/03-01-05-01-01-01-01.wav",
        "sad_sample.wav": "https://zenodo.org/records/8381238/files/03-01-04-01-01-01-01.wav"
    }
    
    # Download each sample file
    for filename, url in sample_files.items():
        destination = os.path.join("demo_files", filename)
        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, destination)
            print(f"✓ Downloaded {filename}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            # Create a dummy file with a tone if download fails
            create_dummy_audio_file(destination)
            print(f"Created placeholder file for {filename}")

def create_dummy_audio_file(filepath, duration=3, sample_rate=22050):
    """Create a dummy audio file with a simple tone"""
    import soundfile as sf
    
    # Generate a simple tone (different for each emotion type)
    if "happy" in filepath:
        # Higher frequency for happy
        t = np.linspace(0, duration, int(duration * sample_rate), False)
        tone = 0.5 * np.sin(2 * np.pi * 440 * t)  # A4 note
        tone += 0.3 * np.sin(2 * np.pi * 880 * t)  # Higher harmonics
    elif "angry" in filepath:
        # Lower frequency with distortion for angry
        t = np.linspace(0, duration, int(duration * sample_rate), False)
        tone = 0.5 * np.sin(2 * np.pi * 220 * t)  # A3 note
        tone = np.clip(tone * 1.5, -1, 1)  # Add some distortion
    elif "sad" in filepath:
        # Lower frequency with slower attack for sad
        t = np.linspace(0, duration, int(duration * sample_rate), False)
        tone = 0.5 * np.sin(2 * np.pi * 196 * t)  # G3 note
        # Add envelope
        envelope = np.linspace(0, 1, int(sample_rate * 0.5))
        envelope = np.concatenate([envelope, np.ones(len(t) - len(envelope))])
        tone = tone * envelope[:len(tone)]
    else:
        # Default tone
        t = np.linspace(0, duration, int(duration * sample_rate), False)
        tone = 0.5 * np.sin(2 * np.pi * 330 * t)  # E4 note
    
    # Apply fade-out
    fade_samples = int(0.1 * sample_rate)
    fade_out = np.linspace(1, 0, fade_samples)
    tone[-fade_samples:] *= fade_out
    
    # Write to file
    sf.write(filepath, tone, sample_rate)

def create_uploads_directory():
    """Create uploads directory for storing user recordings"""
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    print(f"Created {uploads_dir} directory for storing uploaded audio files")

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import tensorflow
        import librosa
        import streamlit
        import soundfile
        import plotly
        print("✓ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def setup_tensorboard_launcher():
    """Setup TensorBoard launcher"""
    # Create link to latest training logs
    if os.path.exists("logs"):
        print("✓ TensorBoard logs directory exists")
    else:
        os.makedirs("logs", exist_ok=True)
        print("Created logs directory for TensorBoard")

def main():
    """Main setup function"""
    print("\n" + "="*50)
    print("Speech Emotion Classification App Setup")
    print("="*50 + "\n")
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Download sample files
    download_sample_files()
    
    # Create uploads directory
    create_uploads_directory()
    
    # Setup TensorBoard
    setup_tensorboard_launcher()
    
    print("\n" + "="*50)
    print("Setup Complete!")
    print("="*50)
    print("\nTo run the application, execute:")
    print("    streamlit run app.py")
    print("\nTo view TensorBoard visualizations, execute:")
    print("    tensorboard --logdir=logs")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()