#!/usr/bin/env python3
# test_model_manager.py - Test functionality of the ModelManager

import os
import sys
import logging
import tensorflow as tf
import numpy as np
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the ModelManager
try:
    # Prefer absolute import that matches the new package structure
    from src.models.model_manager import ModelManager
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    from models.model_manager import ModelManager

def create_dummy_model(model_type="cnn"):
    """Create a simple dummy model for testing"""
    inputs = tf.keras.Input(shape=(128, 128, 1) if model_type == "cnn" else (193,))
    
    if model_type == "cnn":
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
    else:  # mlp
        x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(7, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def test_model_manager():
    logger.info("Testing ModelManager functionality...")
    
    # Initialize ModelManager
    manager = ModelManager()
    
    # Check for existing models
    existing_models = manager.get_models()
    logger.info(f"Found {len(existing_models)} existing models")
    
    # Create and save a dummy model if none exist
    if len(existing_models) == 0:
        # Create a dummy CNN model
        logger.info("Creating a dummy CNN model for testing...")
        cnn_model = create_dummy_model("cnn")
        
        # Save the model
        model_path = "models/cnn_test_emotion_model.keras"
        cnn_model.save(model_path)
        logger.info(f"Saved dummy CNN model to {model_path}")
        
        # Register the model
        metrics = {
            "accuracy": 0.85,
            "precision": 0.83,
            "recall": 0.82,
            "f1": 0.82,
            "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": "RAVDESS (test)",
            "epochs": 10,
            "batch_size": 32
        }
        
        model_id = manager.register_model(
            model_path=model_path,
            model_type="cnn",
            metrics=metrics,
            description="Dummy CNN model for testing"
        )
        
        logger.info(f"Registered model with ID: {model_id}")
        
        # Create a dummy MLP model
        logger.info("Creating a dummy MLP model for testing...")
        mlp_model = create_dummy_model("mlp")
        
        # Save the model
        model_path = "models/mlp_test_emotion_model.keras"
        mlp_model.save(model_path)
        logger.info(f"Saved dummy MLP model to {model_path}")
        
        # Register the model
        metrics = {
            "accuracy": 0.78,
            "precision": 0.76,
            "recall": 0.75,
            "f1": 0.75,
            "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": "RAVDESS (test)",
            "epochs": 15,
            "batch_size": 32
        }
        
        model_id = manager.register_model(
            model_path=model_path,
            model_type="mlp",
            metrics=metrics,
            description="Dummy MLP model for testing"
        )
        
        logger.info(f"Registered model with ID: {model_id}")
    
    # List all registered models
    models = manager.get_models()
    logger.info(f"Found {len(models)} registered models after registration")
    
    for model in models:
        logger.info(f"Model ID: {model['id']}")
        logger.info(f"  Path: {model['path']}")
        logger.info(f"  Type: {model['type']}")
        logger.info(f"  Created: {model['created']}")
        logger.info(f"  Size: {model.get('size_mb', 'Unknown')} MB")
        logger.info(f"  Metrics: {model.get('metrics', {})}")
        logger.info("")
    
    # Load the latest model
    logger.info("Testing model loading...")
    latest_model = manager.get_latest_model()
    if latest_model:
        logger.info(f"Latest model: {latest_model['id']}")
        model = manager.load_model(model_id=latest_model['id'])
        if model:
            logger.info("Model loaded successfully")
            model.summary()
        else:
            logger.error("Failed to load model")
    else:
        logger.error("No models found")
    
    # Test model type filtering
    logger.info("Testing model type filtering...")
    cnn_models = manager.get_models(model_type="cnn")
    logger.info(f"Found {len(cnn_models)} CNN models")
    
    mlp_models = manager.get_models(model_type="mlp")
    logger.info(f"Found {len(mlp_models)} MLP models")
    
    # Test model evaluation report
    logger.info("Testing model evaluation report retrieval...")
    if latest_model:
        report = manager.get_model_evaluation_report(model_id=latest_model['id'])
        if report:
            logger.info(f"Evaluation report: {report}")
        else:
            logger.info("No evaluation report found")
    
    logger.info("ModelManager tests completed successfully!")

if __name__ == "__main__":
    test_model_manager()
