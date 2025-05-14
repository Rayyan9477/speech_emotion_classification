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
        logging.FileHandler(str(config.Config().paths.logs_dir + "/speech_emotion.log")),
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
            train_size=config.Config().training.train_split,
            val_size=config.Config().training.val_split,
            test_size=config.Config().training.test_split
        )
        
        # Extract features with specified configuration
        logger.info(f"Extracting {args.feature_type} features...")
        feature_config = config.Config().features.mel_spectrogram if args.feature_type == 'mel_spectrogram' else config.Config().features.mfcc
        
        train_features = feature_extractor.process_dataset(
            train_data, feature_type=args.feature_type
        )
        val_features = feature_extractor.process_dataset(
            val_data, feature_type=args.feature_type
        )
        test_features = feature_extractor.process_dataset(
            test_data, feature_type=args.feature_type
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
            num_classes=len(config.Config().training.emotion_labels)
        )        # Build model with appropriate configuration
        if args.model_type == 'mlp':
            model = emotion_model.build_mlp(
                input_shape=train_features['mfcc'][0].shape,
                hidden_layers=config.Config().models.mlp.hidden_layers,
                dropout_rate=config.Config().models.mlp.dropout_rate
            )
        else:  # cnn
            # Define CNN parameters
            params = {
                'learning_rate': config.Config().models.cnn.learning_rate,
                'num_conv_layers': len(config.Config().models.cnn.conv_layers),
                'filters': config.Config().models.cnn.conv_layers,
                'kernel_size': (3, 3),
                'pool_size': (2, 2),
                'num_dense_layers': len(config.Config().models.cnn.dense_layers),
                'dense_units': config.Config().models.cnn.dense_layers,
                'dropout_rate': config.Config().models.cnn.dropout_rate
            }
            
            # Add channel dimension to input shape if needed
            input_shape = train_features['mel_spectrogram'][0].shape
            if len(input_shape) == 2:
                input_shape = (*input_shape, 1)
                
            model = emotion_model.build_cnn(
                input_shape=input_shape,
                params=params
            )
          # Configure trainer
        model_config = config.Config().models.cnn if args.model_type == 'cnn' else config.Config().models.mlp
        trainer = ModelTrainer(
            model=model,
            model_type=args.model_type
        )        # Configure callbacks
        callbacks = emotion_model.get_callbacks(
            patience=args.patience
        )
        
        # Train model
        history = trainer.train(
            X_train=train_features['mel_spectrogram'],
            y_train=train_features['labels'],
            X_val=val_features['mel_spectrogram'],
            y_val=val_features['labels'],
            batch_size=args.batch_size,
            epochs=args.epochs,
            callbacks=callbacks
        )
        
        # Evaluate model
        metrics = trainer.evaluate(
            X_test=test_features['mel_spectrogram'],
            y_test=test_features['labels']
        )
        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = Path(config.Config().paths.models_dir) / f"{args.model_type}_emotion_model_{timestamp}.keras"
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
        model_config = config.Config().models.cnn if model_info['type'] == 'cnn' else config.Config().models.mlp
        trainer = ModelTrainer(
            model=model, 
            model_type=model_info['type'],
            learning_rate=model_config.learning_rate
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
    parser.add_argument('--model-type', choices=config.Config().training.model_types, default='cnn', help='Type of model to use')
    parser.add_argument('--feature-type', choices=['mel_spectrogram', 'mfcc'], default='mel_spectrogram', help='Type of features to extract')
    parser.add_argument('--batch-size', type=int, 
                       default=config.Config().models.cnn.batch_size, help='Training batch size')
    parser.add_argument('--epochs', type=int, 
                       default=config.Config().models.cnn.epochs, help='Number of training epochs')
    parser.add_argument('--patience', type=int, 
                       default=config.Config().models.cnn.early_stopping_patience, help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Load configuration based on model type
    model_config = config.Config().models.cnn if args.model_type == 'cnn' else config.Config().models.mlp
    args.batch_size = args.batch_size or model_config.batch_size
    args.epochs = args.epochs or model_config.epochs
    
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
