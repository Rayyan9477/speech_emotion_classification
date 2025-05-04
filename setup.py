#!/usr/bin/env python3
# setup.py - Prepare demo files and set up the environment for the UI

import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
import logging
from pathlib import Path
import random
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_directories():
    """Create required directories for the application"""
    directories = ["uploads", "demo_files", "assets"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def extract_sample_files():
    """Extract sample files from the dataset for demos"""
    try:
        logger.info("Extracting sample files for demo section...")
        
        # Use the test data saved during model training
        X_test = np.load('results/cnn_X_test.npy')
        y_test = np.load('results/cnn_y_test.npy')
        
        # Emotion labels from the original dataset
        emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        
        # Create synthetic audio samples if real samples aren't found
        # This is a simple way to ensure the demo works even without real audio files
        for emotion in ['happy', 'angry', 'sad']:
            demo_path = f"demo_files/{emotion}_sample.wav"
            
            # Check if we already have this demo file
            if os.path.exists(demo_path):
                logger.info(f"Demo file already exists: {demo_path}")
                continue
                
            try:
                # Try to find samples in the test set
                emotion_idx = emotion_labels.index(emotion)
                emotion_samples = np.where(y_test == emotion_idx)[0]
                
                if len(emotion_samples) > 0:
                    # We found samples for this emotion in the test set
                    # However, we don't have direct access to the raw audio files
                    # So we'll create a synthetic audio file for demonstration purposes
                    sample_idx = emotion_samples[0]
                    
                    # Generate synthetic audio based on white noise with emotion-specific patterns
                    sr = 16000  # Sample rate
                    duration = 3  # 3 seconds
                    
                    # Create base signal (white noise)
                    samples = np.random.normal(0, 0.1, size=sr * duration)
                    
                    # Add emotion-specific modulation
                    if emotion == 'happy':
                        # Higher frequency modulation for happy
                        mod = np.sin(2 * np.pi * 8 * np.arange(sr * duration) / sr)
                        samples = samples + 0.3 * mod
                    elif emotion == 'angry':
                        # Sharper, more erratic modulation for angry
                        mod = np.sin(2 * np.pi * 3 * np.arange(sr * duration) / sr)
                        samples = samples + 0.4 * np.abs(mod)
                    elif emotion == 'sad':
                        # Slower, smoother modulation for sad
                        mod = np.sin(2 * np.pi * 1 * np.arange(sr * duration) / sr)
                        samples = samples + 0.2 * mod
                    
                    # Save as WAV file
                    import soundfile as sf
                    sf.write(demo_path, samples, sr)
                    logger.info(f"Created synthetic {emotion} demo file: {demo_path}")
                else:
                    # If no samples found, create a simple tone
                    create_synthetic_audio(emotion, demo_path)
            except Exception as e:
                logger.warning(f"Could not extract {emotion} sample: {e}")
                create_synthetic_audio(emotion, demo_path)
                
        logger.info("Sample files extraction completed")
    except Exception as e:
        logger.error(f"Error extracting sample files: {e}")
        logger.info("Creating synthetic audio samples as fallback...")
        create_synthetic_audio("happy", "demo_files/happy_sample.wav")
        create_synthetic_audio("angry", "demo_files/angry_sample.wav")
        create_synthetic_audio("sad", "demo_files/sad_sample.wav")

def create_synthetic_audio(emotion, output_path):
    """Create a synthetic audio file for demonstration purposes"""
    sr = 16000  # Sample rate
    duration = 3  # seconds
    
    # Create a simple tone with different characteristics based on emotion
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    if emotion == 'happy':
        # Higher frequency for happy
        frequency = 440  # A4 note
        amplitude_mod = 0.1 * np.sin(2 * np.pi * 2 * t)  # Vibrato
        samples = 0.5 * np.sin(2 * np.pi * frequency * t + amplitude_mod)
    elif emotion == 'angry':
        # More dissonant for angry
        frequency = 220  # A3 note
        noise = 0.1 * np.random.normal(0, 1, size=len(t))
        samples = 0.5 * np.sin(2 * np.pi * frequency * t) + noise
    elif emotion == 'sad':
        # Lower frequency for sad
        frequency = 196  # G3 note
        samples = 0.5 * np.sin(2 * np.pi * frequency * t)
        # Add slow decay
        decay = np.linspace(1, 0.3, len(samples))
        samples = samples * decay
    else:
        # Default
        frequency = 329.63  # E4 note
        samples = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Save as WAV file
    import soundfile as sf
    sf.write(output_path, samples, sr)
    logger.info(f"Created synthetic {emotion} audio: {output_path}")

def create_logo():
    """Create a simple logo for the application if needed"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Circle
        
        # Check if logo already exists
        logo_path = "assets/logo.png"
        if os.path.exists(logo_path):
            logger.info(f"Logo already exists: {logo_path}")
            return
            
        # Create directory if it doesn't exist
        os.makedirs("assets", exist_ok=True)
        
        # Create a simple logo using matplotlib
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_aspect('equal')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        # Background
        ax.add_patch(Rectangle((0, 0), 10, 10, facecolor='#4527A0', alpha=0.2))
        
        # Speech bubble
        ax.add_patch(Rectangle((2, 2), 6, 4, facecolor='#7E57C2', alpha=0.7, 
                              edgecolor='white', linewidth=2, zorder=2))
        # Speech bubble pointer
        ax.plot([3, 2, 4], [2, 1, 2], color='white', linewidth=2, zorder=2)
        
        # Sound waves
        for i in range(1, 4):
            circle = Circle((5, 5), i, fill=False, edgecolor='white', 
                           linewidth=2, alpha=0.7-i*0.15, zorder=3)
            ax.add_patch(circle)
        
        # Add text
        ax.text(5, 5, "🎭", fontsize=24, ha='center', va='center', zorder=4)
        ax.text(5, 8, "Speech Emotion", fontsize=14, ha='center', va='center', 
               color='#4527A0', fontweight='bold', zorder=4)
        
        # Remove axes
        ax.axis('off')
        
        # Save the logo
        plt.savefig(logo_path, dpi=300, bbox_inches='tight', transparent=True)
        plt.close()
        
        logger.info(f"Created logo: {logo_path}")
    except Exception as e:
        logger.warning(f"Could not create logo: {e}")

def main():
    parser = argparse.ArgumentParser(description='Setup the Speech Emotion Analysis UI')
    parser.add_argument('--force', action='store_true', help='Force recreation of sample files even if they exist')
    args = parser.parse_args()
    
    try:
        logger.info("Setting up Speech Emotion Analysis UI...")
        
        # Create required directories
        setup_directories()
        
        # Create logo
        create_logo()
        
        # Extract sample files if they don't exist or if force flag is set
        if args.force or not all(os.path.exists(f"demo_files/{emotion}_sample.wav") 
                               for emotion in ['happy', 'angry', 'sad']):
            extract_sample_files()
        else:
            logger.info("Sample files already exist. Use --force to recreate them.")
        
        logger.info("Setup completed successfully!")
        print("\n" + "="*50)
        print(" Speech Emotion Analysis UI Setup Complete! ")
        print("="*50)
        print("\nRun the application with:\n")
        print("    streamlit run app.py\n")
        
    except Exception as e:
        logger.error(f"Error during setup: {e}")
        print("\nSetup failed. See log for details.")

if __name__ == "__main__":
    main()