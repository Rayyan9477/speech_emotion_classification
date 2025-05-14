#!/usr/bin/env python3
"""Main entry point for the Speech Emotion Classification system."""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Import core configuration
from src.core import config

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(config.LOGS_DIR / "speech_emotion.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import monkey patch first to fix TensorFlow issues
from src.utils.monkey_patch import monkeypatch
monkeypatch()

# Try to import TensorFlow with error handling
try:
    import tensorflow as tf
    tensorflow_available = True
    logger.info(f"TensorFlow version: {tf.__version__}")
except ImportError as e:
    tensorflow_available = False
    tensorflow_error = str(e)
    logger.error(f"TensorFlow not available: {tensorflow_error}")

# Import module dependencies
import numpy as np
import tensorflow as tf

# Import our modules
from src.data.data_loader import DataLoader
from src.features.feature_extractor import FeatureExtractor
from src.models.emotion_model import EmotionModel
from src.models.trainer import ModelTrainer
from src.models.model_manager import ModelManager

def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    import numpy as np
    np.random.seed(seed)
    if tensorflow_available:
        tf.random.set_seed(seed)

def train_model(args):
    """Train a new model with the specified configuration."""
    try:
        # Set random seed for reproducibility
        set_seeds(42)

        # Initialize components
        data_loader = DataLoader()
        feature_extractor = FeatureExtractor()
        model_manager = ModelManager()
        
        # Load and prepare data
        logger.info("Loading dataset...")
        dataset = data_loader.load_dataset()
        train_data, val_data, test_data = data_loader.split_dataset(
            validation_split=config.CNN_CONFIG['validation_split']
        )
        
        # Extract features with specified configuration
        logger.info(f"Extracting {args.feature_type} features...")
        feature_config = config.MEL_CONFIG if args.feature_type == 'mel_spectrogram' else config.MFCC_CONFIG
        
        train_features = feature_extractor.process_dataset(
            train_data, feature_type=args.feature_type, **feature_config
        )
        val_features = feature_extractor.process_dataset(
            val_data, feature_type=args.feature_type, **feature_config
        )
        test_features = feature_extractor.process_dataset(
            test_data, feature_type=args.feature_type, **feature_config
        )
        
        # Normalize features
        train_features = feature_extractor.normalize_features(
            train_features, feature_type=args.feature_type
        )
        val_features = feature_extractor.normalize_features(
            val_features, feature_type=args.feature_type, fit=False
        )
        test_features = feature_extractor.normalize_features(
            test_features, feature_type=args.feature_type, fit=False
        )
        
        # Create and train model
        logger.info(f"Creating {args.model_type.upper()} model...")
        emotion_model = EmotionModel(
            num_classes=len(config.EMOTION_LABELS), 
            model_type=args.model_type
        )
          # Build model with appropriate configuration
        if args.model_type == 'mlp':
            model = emotion_model.build_mlp(
                input_shape=train_features['mfcc'][0].shape,
                hidden_layers=config.MLP_CONFIG['hidden_layers'],
                dropout_rate=config.MLP_CONFIG['dropout_rate']
            )
        else:  # cnn
            model = emotion_model.build_cnn(
                input_shape=train_features['spectrogram'][0].shape,
                conv_layers=config.CNN_CONFIG['conv_layers'],
                dense_layers=config.CNN_CONFIG['dense_layers'],
                dropout_rate=config.CNN_CONFIG['dropout_rate']
            )
        
        # Configure trainer
        model_config = config.CNN_CONFIG if args.model_type == 'cnn' else config.MLP_CONFIG
        trainer = ModelTrainer(
            model=model,
            model_type=args.model_type,
            learning_rate=model_config['learning_rate']
        )
        
        # Configure callbacks
        callbacks = emotion_model.get_callbacks(
            patience=args.patience,
            reduce_lr_patience=model_config['reduce_lr_patience']
        )
        
        # Train model
        history = trainer.train(
            X_train=train_features[args.feature_type],
            y_train=train_features['labels'],
            X_val=val_features[args.feature_type],
            y_val=val_features['labels'],
            batch_size=args.batch_size,
            epochs=args.epochs,
            callbacks=callbacks
        )
        
        # Evaluate model
        metrics = trainer.evaluate(
            X_test=test_features[args.feature_type],
            y_test=test_features['labels']
        )
          # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = config.MODELS_DIR / f"{args.model_type}_emotion_model_{timestamp}.keras"
        backup_path = model_path.with_suffix('.h5')
        
        trainer.save_model(model_path)
        trainer.save_model(backup_path)  # Save backup format
        
        # Register model with full metadata
        metrics['trained_on'] = timestamp
        metrics['feature_type'] = args.feature_type
        metrics['feature_config'] = feature_config
        metrics['model_config'] = model_config
        metrics['num_params'] = model.count_params()
        metrics['training_time'] = trainer.training_time
        
        model_manager.register_model(
            model_path=str(model_path),
            model_type=args.model_type,
            metrics=metrics,
            description=f"Trained {args.model_type.upper()} model using {args.feature_type} features"
        )
        
        # Save training history
        model_manager.save_training_history(
            history=history,
            model_id=f"{args.model_type}_{timestamp}",
            model_type=args.model_type
        )
        
        # Save test data for future evaluation
        model_manager.save_test_data(
            X_test=test_features[args.feature_type],
            y_test=test_features['labels'],
            model_type=args.model_type
        )
        
        # Save feature extraction info
        feature_info = {
            'feature_type': args.feature_type,
            'config': feature_config,
            'normalization_params': feature_extractor.get_normalization_params()
        }
        model_manager.save_feature_info(feature_info, model_path=str(model_path))
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

def evaluate_model(args):
    """Evaluate an existing model."""
    try:
        # Load model and metadata
        model_manager = ModelManager()
        model = model_manager.load_model(model_id=args.model_id)
        model_info = model_manager.get_model_by_id(args.model_id)
        
        if model is None:
            logger.error(f"Could not load model with ID: {args.model_id}")
            return
        
        if not model_info:
            logger.error(f"Could not find model metadata for ID: {args.model_id}")
            return
        
        # Load test data from results directory
        test_data_path = config.RESULTS_DIR / f"{model_info['type']}"
        try:
            X_test = np.load(test_data_path / "X_test.npy")
            y_test = np.load(test_data_path / "y_test.npy")
        except FileNotFoundError:
            logger.error(f"Test data not found in {test_data_path}")
            logger.info("Please run training first to generate test data")
            return
        
        # Create trainer and evaluate
        model_config = config.CNN_CONFIG if model_info['type'] == 'cnn' else config.MLP_CONFIG
        trainer = ModelTrainer(
            model=model, 
            model_type=model_info['type'],
            learning_rate=model_config['learning_rate']
        )
        
        # Evaluate model
        start_time = time.time()
        metrics = trainer.evaluate(X_test, y_test)
        eval_time = time.time() - start_time
        
        # Add evaluation metadata
        metrics.update({
            'evaluated_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'eval_time': eval_time,
            'num_test_samples': len(y_test)
        })
        
        # Save updated metrics
        model_manager.save_model_metrics(
            model_path=model_info['path'],
            metrics=metrics
        )
        
        # Log results
        logger.info(f"Evaluation metrics for model {args.model_id}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"{metric}: {value:.4f}")
            else:
                logger.info(f"{metric}: {value}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        logger.debug("Stack trace:", exc_info=True)
        raise

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Speech Emotion Classification System")
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate an existing model')
    parser.add_argument('--model-id', help='Model ID for evaluation')
    parser.add_argument('--model-type', choices=config.MODEL_TYPES, default='cnn', help='Type of model to use')
    parser.add_argument('--feature-type', choices=config.FEATURE_TYPES, default='mel_spectrogram', help='Type of features to extract')
    parser.add_argument('--batch-size', type=int, 
                       default=config.CNN_CONFIG['batch_size'], help='Training batch size')
    parser.add_argument('--epochs', type=int, 
                       default=config.CNN_CONFIG['epochs'], help='Number of training epochs')
    parser.add_argument('--patience', type=int, 
                       default=config.CNN_CONFIG['early_stopping_patience'], help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Load configuration based on model type
    model_config = config.CNN_CONFIG if args.model_type == 'cnn' else config.MLP_CONFIG
    args.batch_size = args.batch_size or model_config['batch_size']
    args.epochs = args.epochs or model_config['epochs']
    
    if args.train:
        train_model(args)
    elif args.evaluate:
        if args.model_id is None:
            parser.error("--model-id is required for evaluation")
        evaluate_model(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)
