#!/usr/bin/env python3
"""Training pipeline utilities (Streamlit-integrated; no CLI parsing)."""

import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, Any

# Import core configuration
from src.core import config

# Setup logging configuration
try:
    cfg = config.Config()
    log_dir = Path(cfg.paths.logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "speech_emotion.log"
except Exception:
    # Fallback to current directory if config paths fail
    log_file = Path("speech_emotion.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_file)),
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

@dataclass
class TrainArgs:
    model_type: str = "cnn"  # or 'mlp'
    feature_type: str = "mel_spectrogram"  # or 'mfcc'
    batch_size: int = config.Config().models.cnn.batch_size
    epochs: int = config.Config().models.cnn.epochs
    patience: int = config.Config().models.cnn.early_stopping_patience


def train_model(args: TrainArgs) -> Tuple[str, Dict[str, Any]]:
    """Train a new model.

    Returns
    -------
    (model_path, metrics): Tuple containing the saved model path and metrics dict.
    """
    try:
        set_seeds(42)

        cfg_local = config.Config()  # fresh instance in case of dynamic config
        data_loader = DataLoader()
        feature_extractor = FeatureExtractor()
        model_manager = ModelManager()

        # Load and split dataset
        logger.info("Loading dataset...")
        _ = data_loader.load_dataset()
        train_data, val_data, test_data = data_loader.split_dataset(
            train_size=cfg_local.training.train_split,
            val_size=cfg_local.training.val_split,
            test_size=cfg_local.training.test_split
        )

        # Feature extraction
        logger.info("Extracting %s features...", args.feature_type)
        feature_config = (cfg_local.features.mel_spectrogram
                          if args.feature_type == 'mel_spectrogram'
                          else cfg_local.features.mfcc)

        train_features = feature_extractor.process_dataset(train_data, feature_type=args.feature_type)
        val_features = feature_extractor.process_dataset(val_data, feature_type=args.feature_type)
        test_features = feature_extractor.process_dataset(test_data, feature_type=args.feature_type)

        # Normalize (fit on train only)
        train_features = feature_extractor.normalize_features(train_features, feature_type=args.feature_type)
        val_features = feature_extractor.normalize_features(val_features, feature_type=args.feature_type, fit=False)
        test_features = feature_extractor.normalize_features(test_features, feature_type=args.feature_type, fit=False)

        # Build model
        logger.info("Creating %s model...", args.model_type.upper())
        emotion_model = EmotionModel(num_classes=len(cfg_local.training.emotion_labels))

        if args.model_type == 'mlp':
            mlp_hidden = cfg_local.models.mlp.hidden_layers
            mlp_params = {
                'learning_rate': cfg_local.models.mlp.learning_rate,
                'num_layers': len(mlp_hidden),
                'units': mlp_hidden,
                'dropout_rate': cfg_local.models.mlp.dropout_rate,
            }
            model = emotion_model.build_mlp(
                input_shape=train_features['mfcc'][0].shape,
                params=mlp_params
            )
        else:  # cnn
            params = {
                'learning_rate': cfg_local.models.cnn.learning_rate,
                'num_conv_layers': len(cfg_local.models.cnn.conv_layers),
                'filters': cfg_local.models.cnn.conv_layers,
                'kernel_size': (3, 3),
                'pool_size': (2, 2),
                'num_dense_layers': len(cfg_local.models.cnn.dense_layers),
                'dense_units': cfg_local.models.cnn.dense_layers,
                'dropout_rate': cfg_local.models.cnn.dropout_rate
            }
            input_shape = train_features['mel_spectrogram'][0].shape
            if len(input_shape) == 2:  # add channel dim if missing
                input_shape = (*input_shape, 1)
            model = emotion_model.build_cnn(input_shape=input_shape, params=params)

        model_config = cfg_local.models.cnn if args.model_type == 'cnn' else cfg_local.models.mlp
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

        # Evaluate on test
        metrics = trainer.evaluate(
            X_test=test_features[feature_key],
            y_test=test_features['labels'],
            emotion_labels=cfg_local.training.emotion_labels
        )

        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = Path(cfg_local.paths.models_dir) / f"{args.model_type}_emotion_model_{timestamp}.keras"
        backup_path = model_path.with_suffix('.h5')

        trainer.save_model(model_path)
        try:
            trainer.save_model(backup_path)  # optional backup format
        except Exception:
            pass

        # Register model and save metadata
        metrics.update({
            'trained_on': timestamp,
            'feature_type': args.feature_type,
            'feature_config': feature_config,
            'model_config': model_config.__dict__ if hasattr(model_config, '__dict__') else str(model_config),
            'num_params': model.count_params(),
            'training_time': getattr(trainer, 'training_time', None),
        })

        model_id = model_manager.register_model(
            model_path=str(model_path),
            model_type=args.model_type,
            metrics=metrics,
            description=f"Trained {args.model_type.upper()} model using {args.feature_type} features"
        )

        # Persist artifacts
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

        feature_info = {
            'feature_type': args.feature_type,
            'config': feature_config,
            'normalization_params': feature_extractor.get_normalization_params()
        }
        model_manager.save_feature_info(feature_info, model_path=str(model_path))

        logger.info("Training completed successfully: %s", model_path)
        return str(model_path), metrics
    except Exception as e:
        logger.error("Error during training: %s", e, exc_info=True)
        raise
    
