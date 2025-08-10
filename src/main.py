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

# Try to import TensorFlow with error handling (keep tf symbol)
try:
    import tensorflow as tf
    tensorflow_available = True
    logger.info("TensorFlow version: %s", tf.__version__)
except Exception as e:
    tf = None
    tensorflow_available = False
    logger.error(f"TensorFlow not available: {e}")

# Import module dependencies
import numpy as np

# Import our modules
from src.data.data_loader import DataLoader
from src.features.feature_extractor import FeatureExtractor
from src.models.emotion_model import EmotionModel
from src.models.trainer import ModelTrainer
from src.models.model_manager import ModelManager

def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    if tensorflow_available and tf is not None:
        tf.random.set_seed(seed)

def train_model(args):
    """Train a new model with the specified configuration."""
    try:
        set_seeds(42)

        data_loader = DataLoader()
        feature_extractor = FeatureExtractor()
        model_manager = ModelManager()

        # Load and split dataset
        logger.info("Loading dataset...")
        _ = data_loader.load_dataset()
        train_data, val_data, test_data = data_loader.split_dataset(
            train_size=config.Config().training.train_split,
            val_size=config.Config().training.val_split,
            test_size=config.Config().training.test_split
        )

        # Feature extraction
        logger.info(f"Extracting {args.feature_type} features...")
        feature_config = config.Config().features.mel_spectrogram if args.feature_type == 'mel_spectrogram' else config.Config().features.mfcc

        train_features = feature_extractor.process_dataset(train_data, feature_type=args.feature_type)
        val_features = feature_extractor.process_dataset(val_data, feature_type=args.feature_type)
        test_features = feature_extractor.process_dataset(test_data, feature_type=args.feature_type)

        # Normalize
        train_features = feature_extractor.normalize_features(train_features, feature_type=args.feature_type)
        val_features = feature_extractor.normalize_features(val_features, feature_type=args.feature_type, fit=False)
        test_features = feature_extractor.normalize_features(test_features, feature_type=args.feature_type, fit=False)

        # Build model
        logger.info(f"Creating {args.model_type.upper()} model...")
        emotion_model = EmotionModel(num_classes=len(config.Config().training.emotion_labels))

        if args.model_type == 'mlp':
            mlp_hidden = config.Config().models.mlp.hidden_layers
            mlp_params = {
                'learning_rate': config.Config().models.mlp.learning_rate,
                'num_layers': len(mlp_hidden),
                'units': mlp_hidden,
                'dropout_rate': config.Config().models.mlp.dropout_rate,
            }
            model = emotion_model.build_mlp(
                input_shape=train_features['mfcc'][0].shape,
                params=mlp_params
            )
        else:  # cnn
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
            input_shape = train_features['mel_spectrogram'][0].shape
            if len(input_shape) == 2:
                input_shape = (*input_shape, 1)
            model = emotion_model.build_cnn(input_shape=input_shape, params=params)

        model_config = config.Config().models.cnn if args.model_type == 'cnn' else config.Config().models.mlp
        trainer = ModelTrainer(model=model, model_type=args.model_type)
        callbacks = emotion_model.get_callbacks(patience=args.patience)

        feature_key = 'mel_spectrogram' if args.feature_type == 'mel_spectrogram' else 'mfcc'
        history = trainer.train(
            X_train=train_features[feature_key],
            y_train=train_features['labels'],
            X_val=val_features[feature_key],
            y_val=val_features['labels'],
            batch_size=args.batch_size,
            epochs=args.epochs,
            callbacks=callbacks
        )

        # Evaluate on test with proper feature key
        metrics = trainer.evaluate(
            X_test=test_features[feature_key],
            y_test=test_features['labels'],
            emotion_labels=config.Config().training.emotion_labels
        )

        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = Path(config.Config().paths.models_dir) / f"{args.model_type}_emotion_model_{timestamp}.keras"
        backup_path = model_path.with_suffix('.h5')

        trainer.save_model(model_path)
        try:
            trainer.save_model(backup_path)  # optional backup format
        except Exception:
            pass

        # Register model and save metadata properly
        metrics.update({
            'trained_on': timestamp,
            'feature_type': args.feature_type,
            'feature_config': feature_config,
            'model_config': model_config.__dict__ if hasattr(model_config, '__dict__') else str(model_config),
            'num_params': model.count_params(),
            'training_time': trainer.training_time,
        })

        model_id = model_manager.register_model(
            model_path=str(model_path),
            model_type=args.model_type,
            metrics=metrics,
            description=f"Trained {args.model_type.upper()} model using {args.feature_type} features"
        )

        # Save training history and test data using the same ID/key
        model_manager.save_training_history(
            history=history,
            model_id=model_id,
            model_type=args.model_type
        )
        model_manager.save_test_data(
            X_test=test_features[feature_key],
            y_test=test_features['labels'],
            model_type=args.model_type
        )

        # Save feature extraction info sidecar
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
        model_manager = ModelManager()
        model = model_manager.load_model(model_id=args.model_id)
        model_info = model_manager.get_model_by_id(args.model_id)

        if model is None or not model_info:
            logger.error(f"Could not load model or metadata for ID: {args.model_id}")
            return

        # Load test data from results directory based on detected type
        results_dir = Path(config.Config().paths.results_dir)
        try:
            X_test = np.load(results_dir / f"{model_info['type']}_X_test.npy")
            y_test = np.load(results_dir / f"{model_info['type']}_y_test.npy")
        except FileNotFoundError:
            logger.error(f"Test data not found in {results_dir}")
            logger.info("Please run training first to generate test data")
            return

        trainer = ModelTrainer(model=model, model_type=model_info['type'])

        start_time = time.time()
        metrics = trainer.evaluate(X_test, y_test, emotion_labels=config.Config().training.emotion_labels)
        eval_time = time.time() - start_time
        metrics.update({
            'evaluated_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'eval_time': eval_time,
            'num_test_samples': len(y_test)
        })

        model_manager.save_model_metrics(model_path=model_info['path'], metrics=metrics)

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
