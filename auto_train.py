#!/usr/bin/env python3
"""
Automatic training script for emotion recognition model.
This script trains a basic CNN model using demo files when no trained model is available.
"""

import sys
import os
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import project modules
from src.core.config import Config
from src.features.feature_extractor import FeatureExtractor
from src.models.emotion_model import EmotionModel
from src.models.model_manager import ModelManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_demo_dataset():
    """Create a simple demo dataset from available audio files."""
    try:
        config = Config()
        feature_extractor = FeatureExtractor()

        # Define emotion mapping for demo files
        emotion_mapping = {
            'happy': 4,  # happy
            'sad': 5,    # sad
            'angry': 0,  # angry
            'neutral': 1,  # neutral
            'calm': 2,    # calm
            'fearful': 3, # fearful
            'disgust': 6, # disgust
            'surprised': 7 # surprised
        }

        features = []
        labels = []

        demo_dir = Path('demo_files')
        if not demo_dir.exists():
            logger.error("Demo files directory not found")
            return None, None

        # Process each demo file
        for audio_file in demo_dir.glob('*.wav'):
            emotion_name = audio_file.stem.split('_')[0].lower()

            if emotion_name in emotion_mapping:
                try:
                    # Extract features
                    file_features = feature_extractor.process_audio_file(str(audio_file))
                    if 'mel_spectrogram' in file_features:
                        features.append(file_features['mel_spectrogram'])
                        labels.append(emotion_mapping[emotion_name])
                        logger.info(f"Processed {audio_file.name} as {emotion_name}")
                except Exception as e:
                    logger.warning(f"Failed to process {audio_file.name}: {e}")

        if len(features) == 0:
            logger.error("No valid audio files found for training")
            return None, None

        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)

        logger.info(f"Created dataset with {len(X)} samples")
        return X, y

    except Exception as e:
        logger.error(f"Error creating demo dataset: {e}")
        return None, None

def train_basic_model():
    """Train a basic CNN model for emotion recognition."""
    try:
        logger.info("Starting automatic model training...")

        # Create dataset
        X, y = create_demo_dataset()
        if X is None or y is None:
            logger.error("Failed to create training dataset")
            return None

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Create model
        config = Config()
        emotion_model = EmotionModel(num_classes=len(config.training.emotion_labels))

        # Build CNN model
        input_shape = X_train[0].shape
        if len(input_shape) == 2:
            input_shape = (*input_shape, 1)

        model_params = {
            'learning_rate': 0.001,
            'num_conv_layers': 2,
            'filters': [32, 64],
            'kernel_size': (3, 3),
            'pool_size': (2, 2),
            'num_dense_layers': 1,
            'dense_units': [128],
            'dropout_rate': 0.3
        }

        model = emotion_model.build_cnn(input_shape=input_shape, params=model_params)

        # Train model
        logger.info("Training model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=20,
            batch_size=8,
            verbose=1
        )

        # Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        logger.info(".4f")

        # Save model
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)

        model_path = models_dir / 'cnn_emotion_model_auto.keras'
        model.save(str(model_path))

        # Register model
        model_manager = ModelManager()
        model_id = model_manager.register_model(
            model_path=str(model_path),
            model_type='cnn',
            metrics={
                'accuracy': float(test_accuracy),
                'loss': float(test_loss),
                'feature_type': 'mel_spectrogram'
            },
            description='Automatically trained CNN model using demo files'
        )

        logger.info(f"Model saved to {model_path}")
        logger.info(f"Model registered with ID: {model_id}")

        return str(model_path)

    except Exception as e:
        logger.error(f"Error in automatic training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    print("🚀 Starting automatic model training...")
    model_path = train_basic_model()
    if model_path:
        print(f"✅ Model trained and saved to: {model_path}")
    else:
        print("❌ Automatic training failed")
        sys.exit(1)
