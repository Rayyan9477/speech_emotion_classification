#!/usr/bin/env python3
"""
train_cnn_model.py - Train and evaluate the CNN model for speech emotion classification
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train_cnn_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the src directory to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(project_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Apply monkey patch to fix TensorFlow overflow issues before importing TensorFlow
try:
    from src.utils.monkey_patch import monkeypatch
    if monkeypatch():
        logger.info("Successfully applied TensorFlow monkey patch")
    else:
        logger.warning("Failed to apply TensorFlow monkey patch")
except ImportError:
    logger.warning("Could not import monkey_patch module, some TensorFlow operations may fail")

# Import TensorFlow after monkey patching
import tensorflow as tf

# Import our modules
from src.models.emotion_model import EmotionModel
from src.models.trainer import ModelTrainer
from src.models.model_manager import ModelManager

# Constants
MODEL_TYPE = 'cnn'
NUM_CLASSES = 7
MODEL_DIR = 'models'
RESULTS_DIR = 'results'
LOGS_DIR = 'logs'
EPOCHS = 50
BATCH_SIZE = 32
PATIENCE = 10  # For early stopping
EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust']

def load_features(feature_dir='features'):
    """
    Load pre-processed features for training.
    
    For demonstration, we'll generate random data, but in a real application,
    this would load from your preprocessed feature files.
    """
    # In a real application, load your actual feature data
    try:
        # Try loading saved feature data if it exists
        X_train = np.load(os.path.join(feature_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(feature_dir, 'y_train.npy'))
        X_val = np.load(os.path.join(feature_dir, 'X_val.npy'))
        y_val = np.load(os.path.join(feature_dir, 'y_val.npy'))
        X_test = np.load(os.path.join(feature_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(feature_dir, 'y_test.npy'))
        
        logger.info(f"Loaded feature data from {feature_dir}")
        
    except FileNotFoundError:
        # Generate dummy data for demonstration
        logger.warning("Feature files not found. Generating random data for demonstration.")
        
        # For CNN models: (samples, height, width, channels)
        X_train = np.random.random((500, 128, 128, 1)).astype(np.float32)
        y_train = np.random.randint(0, NUM_CLASSES, size=(500,), dtype=np.int32)
        
        X_val = np.random.random((100, 128, 128, 1)).astype(np.float32)
        y_val = np.random.randint(0, NUM_CLASSES, size=(100,), dtype=np.int32)
        
        X_test = np.random.random((100, 128, 128, 1)).astype(np.float32)
        y_test = np.random.randint(0, NUM_CLASSES, size=(100,), dtype=np.int32)
        
        # Save the dummy data for consistency
        os.makedirs(feature_dir, exist_ok=True)
        np.save(os.path.join(feature_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(feature_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(feature_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(feature_dir, 'y_val.npy'), y_val)
        np.save(os.path.join(feature_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(feature_dir, 'y_test.npy'), y_test)
    
    # Log shapes for verification
    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_model():
    """Train the CNN model"""
    # Create directories
    for directory in [MODEL_DIR, RESULTS_DIR, LOGS_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # Load preprocessed features
    X_train, y_train, X_val, y_val, X_test, y_test = load_features()
    
    # Create model
    logger.info(f"Creating {MODEL_TYPE.upper()} model with {NUM_CLASSES} classes")
    emotion_model = EmotionModel(num_classes=NUM_CLASSES)
    
    if MODEL_TYPE == 'cnn':
        # Get input shape from the data
        input_shape = X_train.shape[1:]  # (height, width, channels)
        model = emotion_model.build_cnn(input_shape=input_shape)
    else:
        raise ValueError(f"Model type {MODEL_TYPE} not supported. Only 'cnn' is implemented in this script.")
    
    # Create trainer
    trainer = ModelTrainer(model, model_type=MODEL_TYPE)
    
    # Create callbacks
    checkpoint_path = os.path.join(MODEL_DIR, f"{MODEL_TYPE}_best_model.keras")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    # Add TensorBoard callback
    log_dir = os.path.join(LOGS_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True
    )
    
    callbacks = [early_stopping, checkpoint, tensorboard_callback]
    
    # Train model
    logger.info(f"Starting {MODEL_TYPE.upper()} model training for {EPOCHS} epochs")
    start_time = time.time()
    
    try:
        history = trainer.train(
            X_train, y_train,
            X_val, y_val,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=callbacks
        )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        # Evaluate model
        logger.info("Evaluating model on test data")
        metrics = trainer.evaluate(X_test, y_test, emotion_labels=EMOTION_LABELS)
        
        # Save model with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(MODEL_DIR, f"{MODEL_TYPE}_emotion_model_{timestamp}.keras")
        saved_path = trainer.save_model(model_path)
        
        # Register model with metrics
        manager = ModelManager(
            models_dir=MODEL_DIR,
            results_dir=RESULTS_DIR,
            logs_dir=LOGS_DIR
        )
        
        model_id = manager.register_model(
            model_path=saved_path,
            model_type=MODEL_TYPE,
            metrics=metrics,
            description=f"{MODEL_TYPE.upper()} model trained on speech emotion dataset with {metrics['accuracy']:.4f} accuracy"
        )
        
        # Save training history and test data
        manager.save_training_history(history, model_id, MODEL_TYPE)
        manager.save_test_data(X_test, y_test, MODEL_TYPE)
        
        # Log success
        logger.info(f"Model saved to {saved_path} and registered with ID: {model_id}")
        logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
        
        return model_id, metrics
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

if __name__ == "__main__":
    try:
        # Check for TensorFlow GPU support
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"TensorFlow detected {len(gpus)} GPU(s): {gpus}")
            # Allow memory growth to avoid taking all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            logger.warning("No GPU detected. Training will be slower on CPU.")
        
        # Train and evaluate model
        model_id, metrics = train_model()
        
        if model_id:
            logger.info("===== Training Summary =====")
            logger.info(f"Model ID: {model_id}")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"F1 Score: {metrics['f1_avg']:.4f}")
            logger.info("============================")
            sys.exit(0)
        else:
            logger.error("Training failed.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
