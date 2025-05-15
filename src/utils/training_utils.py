#!/usr/bin/env python3
"""
training_utils.py - Utility functions for model training
"""

import os
import sys
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path

logger = logging.getLogger(__name__)

def setup_environment():
    """
    Set up the environment for training, including TensorFlow configuration.
    """
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Configure TensorFlow to use memory growth
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s), enabled memory growth")
    except Exception as e:
        logger.warning(f"Error configuring GPU: {e}")
    
    # Apply monkey patch if available
    try:
        from src.utils.monkey_patch import monkeypatch
        if monkeypatch():
            logger.info("Successfully applied TensorFlow monkey patch")
        else:
            logger.warning("Failed to apply TensorFlow monkey patch")
    except ImportError:
        logger.warning("Could not import monkey_patch module")
    
    return True

def generate_dummy_data(num_classes=7, input_shape=(20, 20, 1), train_samples=100, val_samples=30, test_samples=30):
    """
    Generate dummy data for testing and development.
    
    Args:
        num_classes (int): Number of emotion classes
        input_shape (tuple): Shape of input data (height, width, channels)
        train_samples (int): Number of training samples
        val_samples (int): Number of validation samples
        test_samples (int): Number of test samples
        
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    logger.info(f"Generating dummy data with shape {input_shape}")
    
    X_train = np.random.random((train_samples, *input_shape)).astype(np.float32)
    y_train = np.random.randint(0, num_classes, size=(train_samples,), dtype=np.int32)

    X_val = np.random.random((val_samples, *input_shape)).astype(np.float32)
    y_val = np.random.randint(0, num_classes, size=(val_samples,), dtype=np.int32)

    X_test = np.random.random((test_samples, *input_shape)).astype(np.float32)
    y_test = np.random.randint(0, num_classes, size=(test_samples,), dtype=np.int32)
    
    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def create_model_directories():
    """
    Create necessary directories for model training.
    """
    directories = ['models', 'logs', 'results', 'results/reports']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    return True

def save_training_history(history, filename='training_history.png'):
    """
    Save training history as a plot.
    
    Args:
        history: Training history object from model.fit()
        filename (str): Filename to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)
        filepath = os.path.join('results', filename)
        plt.savefig(filepath)
        logger.info(f"Training history plot saved to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving training history plot: {e}")
        return None