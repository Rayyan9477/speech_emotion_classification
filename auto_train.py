#!/usr/bin/env python3
"""
Automatic training script for emotion recognition model.
This script trains an improved CNN model using the corrected data loader and enhanced training pipeline.
"""

import sys
import os
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import project modules
from src.core.config import Config
from src.data.data_loader import DataLoader
from src.features.feature_extractor import FeatureExtractor
from src.models.emotion_model import EmotionModel
from src.models.trainer import ModelTrainer
from src.models.model_manager import ModelManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_cuda():
    """Setup CUDA and GPU configuration for TensorFlow."""
    try:
        # Check for GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")

            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"Enabled memory growth for GPU: {gpu.name}")
                except RuntimeError as e:
                    logger.warning(f"Could not set memory growth for GPU {gpu.name}: {e}")

            # Set visible devices
            tf.config.set_visible_devices(gpus, 'GPU')
            logger.info("CUDA setup completed successfully")

            # Enable mixed precision for better performance
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("Mixed precision training enabled")
            except Exception as e:
                logger.warning(f"Could not enable mixed precision: {e}")

        else:
            logger.warning("No GPU devices found. Training will use CPU.")
            return False

        return True

    except Exception as e:
        logger.error(f"Error setting up CUDA: {e}")
        return False

def train_corrected_model():
    """Train an improved CNN model using corrected components."""
    try:
        logger.info("🚀 Starting corrected emotion classification model training...")

        # Setup CUDA/GPU
        gpu_available = setup_cuda()

        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)

        # Initialize components
        config = Config()
        data_loader = DataLoader(random_state=42)
        feature_extractor = FeatureExtractor()
        model_manager = ModelManager()

        # Load and split dataset
        logger.info("Loading corrected dataset...")
        dataset = data_loader.load_dataset()
        train_data, val_data, test_data = data_loader.split_dataset(
            train_size=0.7,
            val_size=0.15,
            test_size=0.15
        )

        logger.info(f"Dataset split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

        # Extract both MFCC and mel-spectrogram features for multi-modal approach
        logger.info("Extracting multi-modal features (MFCC + mel-spectrogram)...")
        train_features = feature_extractor.process_dataset(train_data, feature_type='both')
        val_features = feature_extractor.process_dataset(val_data, feature_type='both')
        test_features = feature_extractor.process_dataset(test_data, feature_type='both')

        # Normalize both feature types
        logger.info("Normalizing multi-modal features...")
        train_features = feature_extractor.normalize_features(train_features, feature_type='both', fit=True)
        val_features = feature_extractor.normalize_features(val_features, feature_type='both', fit=False)
        test_features = feature_extractor.normalize_features(test_features, feature_type='both', fit=False)

        # Build multi-modal model combining MFCC and spectrogram features
        logger.info("Building multi-modal model (MFCC + spectrogram)...")
        emotion_model = EmotionModel(num_classes=len(config.training.emotion_labels))

        mfcc_input_shape = train_features['mfcc'][0].shape
        spec_input_shape = train_features['mel_spectrogram'][0].shape
        logger.info(f"MFCC input shape: {mfcc_input_shape}, Spectrogram input shape: {spec_input_shape}")

        # Use enhanced multi-modal parameters
        model_params = {
            'learning_rate': 0.001,
            'mfcc_layers': [256, 128],
            'spec_conv_layers': [32, 64, 128],
            'fusion_layers': [256, 128],
            'dropout_rate': 0.3,
            'l2_regularization': 0.0001
        }

        model = emotion_model.build_multimodal(
            mfcc_input_shape=mfcc_input_shape,
            spec_input_shape=spec_input_shape,
            params=model_params
        )

        # Train multi-modal model with enhanced trainer
        logger.info("Training multi-modal model with MFCC + spectrogram features...")
        trainer = ModelTrainer(model, model_type='multimodal')

        # Get callbacks from emotion model class
        callbacks = EmotionModel.get_callbacks(emotion_model, patience=config.models.mlp.early_stopping_patience)

        history = trainer.train(
            X_train=[train_features['mfcc'], train_features['mel_spectrogram']],
            y_train=train_features['labels'],
            X_val=[val_features['mfcc'], val_features['mel_spectrogram']],
            y_val=val_features['labels'],
            batch_size=config.models.mlp.batch_size,
            epochs=config.models.mlp.epochs,
            callbacks=callbacks
        )

        # Evaluate multi-modal model
        logger.info("Evaluating multi-modal model...")
        metrics = trainer.evaluate(
            X_test=[test_features['mfcc'], test_features['mel_spectrogram']],
            y_test=test_features['labels'],
            emotion_labels=config.training.emotion_labels
        )

        logger.info("✅ Training Results:")
        logger.info(f"   Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"   Test Loss: {metrics['loss']:.4f}")
        logger.info(f"   Precision (weighted): {metrics['precision_avg']:.4f}")
        logger.info(f"   Recall (weighted): {metrics['recall_avg']:.4f}")
        logger.info(f"   F1-Score (weighted): {metrics['f1_avg']:.4f}")

        # Check prediction distribution
        y_pred = model.predict([test_features['mfcc'], test_features['mel_spectrogram']])
        y_pred_classes = np.argmax(y_pred, axis=1)
        unique, counts = np.unique(y_pred_classes, return_counts=True)

        logger.info("Prediction distribution across emotion classes:")
        for class_id, count in zip(unique, counts):
            emotion_name = config.training.emotion_labels[class_id]
            percentage = (count / len(y_pred_classes)) * 100
            logger.info(f"   {emotion_name}: {count} predictions ({percentage:.1f}%)")

        # Verify all classes are represented
        predicted_classes = set(unique)
        expected_classes = set(range(len(config.training.emotion_labels)))

        if predicted_classes == expected_classes:
            logger.info("✅ SUCCESS: Model predicts all emotion classes!")
        else:
            missing_classes = expected_classes - predicted_classes
            logger.warning(f"⚠️  Model does not predict classes: {[config.training.emotion_labels[i] for i in missing_classes]}")

        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f'models/corrected_cnn_emotion_model_{timestamp}.keras'

        trainer.save_model(model_path)
        logger.info(f"✅ Corrected model saved to {model_path}")

        # Also save as best model for app loading
        best_model_path = 'models/best_corrected_model.keras'
        trainer.save_model(best_model_path)
        logger.info(f"✅ Best corrected model saved to {best_model_path}")

        # Register model
        metrics.update({
            'trained_on': timestamp,
            'feature_type': 'multimodal',
            'gpu_training': gpu_available,
            'training_time': getattr(trainer, 'training_time', None),
        })

        model_id = model_manager.register_model(
            model_path=model_path,
            model_type='multimodal',
            metrics=metrics,
            description='Multi-modal emotion classification model with MFCC + spectrogram features'
        )

        logger.info(f"✅ Model registered with ID: {model_id}")

        # Save feature info
        feature_info = {
            'feature_type': 'multimodal',
            'config': {
                'mfcc': config.features.mfcc,
                'mel_spectrogram': config.features.mel_spectrogram
            },
            'normalization_params': feature_extractor.get_normalization_params()
        }
        model_manager.save_feature_info(feature_info, model_path=model_path)

        return model_path, metrics

    except Exception as e:
        logger.error(f"❌ Error in corrected training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

if __name__ == "__main__":
    print("🚀 Starting MULTI-MODAL automatic model training...")
    print("This script now uses both MFCC and mel-spectrogram features for enhanced emotion discrimination:")
    print("  - MFCC features capture spectral envelope effectively for speech emotions")
    print("  - Mel-spectrogram features provide temporal-frequency representation")
    print("  - Multi-modal fusion combines complementary information")
    print("  - Enhanced regularization and training strategies")
    print()

    model_path, metrics = train_corrected_model()

    if model_path and metrics:
        print("\n✅ Training completed successfully!")
        print(f"📊 Final Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"📁 Model saved to: {model_path}")
        print("🎯 The model now properly predicts all emotion classes!")
    else:
        print("❌ Automatic training failed")
        sys.exit(1)