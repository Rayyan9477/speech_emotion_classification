#!/usr/bin/env python3
# run_app.py - Run the Speech Emotion Classification App

import os
import subprocess
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the enhanced Speech Emotion Classification App"""
    logger.info("Starting Speech Emotion Classification Application")
    
    # Check if model files exist
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logger.info("Created models directory")
    
    cnn_model_path = os.path.join(model_dir, "cnn_emotion_model.keras")
    if not os.path.exists(cnn_model_path):
        logger.warning("No CNN model found. You may need to train a model first")
        response = input("Would you like to train a basic CNN model now? [y/N]: ")
        if response.lower() == 'y':
            logger.info("Training a basic CNN model...")
            try:
                subprocess.run(["python", "main.py", "--model_type", "cnn", "--epochs", "20"], check=True)
                logger.info("Model training completed")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error training model: {e}")
                logger.error("Continuing to app startup, but model functionality may be limited")
    
    # Check if we have the model registry
    registry_path = os.path.join(model_dir, "model_registry.json")
    if not os.path.exists(registry_path):
        logger.info("Initializing model registry...")
        try:
            from model_manager import ModelManager
            ModelManager()  # This will create the registry file
            logger.info("Model registry initialized")
        except ImportError:
            logger.warning("Could not import ModelManager. Model management features will be limited")
    
    # Launch streamlit app
    logger.info("Launching Streamlit app...")
    cmd = ["streamlit", "run", "app.py", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
    
    try:
        # Run Streamlit with the process output visible
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error launching Streamlit app: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("App terminated by user")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
