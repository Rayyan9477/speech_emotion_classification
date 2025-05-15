import os
import json
import shutil
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Class for managing trained models, including:
    - Model registration and metadata tracking
    - Model selection
    - Training history and evaluation metrics
    """
    
    def __init__(self, models_dir="models", results_dir="results", logs_dir="logs"):
        """Initialize the ModelManager with directory paths"""
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.logs_dir = logs_dir
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Load model registry if it exists
        self.registry_path = os.path.join(self.models_dir, "model_registry.json")
        self.model_registry = self._load_registry()
    
    def _load_registry(self):
        """Load the model registry from disk or create a new one"""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading model registry: {e}")
                return {"models": []}
        else:
            return {"models": []}
    
    def _save_registry(self):
        """Save the model registry to disk"""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.model_registry, f, indent=4)
            logger.info(f"Model registry saved to {self.registry_path}")
        except Exception as e:
            logger.error(f"Error saving model registry: {e}")
    
    def register_model(self, model_path, model_type, metrics=None, description=None):
        """
        Register a model in the registry with its metadata
        
        Args:
            model_path (str): Path to the model file
            model_type (str): Type of model (cnn, mlp, etc)
            metrics (dict): Performance metrics of the model
            description (str): Additional description of the model
            
        Returns:
            str: Model ID
        """
        # Generate a unique ID for the model
        model_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get file size in MB
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        # Create model entry
        model_entry = {
            "id": model_id,
            "path": model_path,
            "type": model_type,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "size_mb": round(size_mb, 2),
            "metrics": metrics or {},
            "description": description or f"{model_type.upper()} model trained on RAVDESS dataset"
        }
        
        # Add to registry
        self.model_registry["models"].append(model_entry)
        
        # Save updated registry
        self._save_registry()
        
        logger.info(f"Registered model {model_id} in registry")
        return model_id
    
    def get_models(self, model_type=None):
        """
        Get all registered models, optionally filtered by type
        
        Args:
            model_type (str, optional): Filter models by type
            
        Returns:
            list: List of model entries
        """
        # Scan for new models first
        self._scan_for_new_models()
        
        # Filter by type if specified
        if (model_type):
            return [m for m in self.model_registry["models"] if m["type"].lower() == model_type.lower()]
        return self.model_registry["models"]
    
    def get_best_model(self, metric="accuracy", model_type=None):
        """
        Get the best model based on a specific metric
        
        Args:
            metric (str): Metric to use for comparison
            model_type (str): Filter models by type
            
        Returns:
            dict: Best model entry
        """
        models = self.get_models(model_type)
        if not models:
            logger.warning(f"No models found for type {model_type}")
            return None
        
        # Filter models that have the specified metric
        valid_models = [m for m in models if "metrics" in m and metric in m["metrics"]]
        if not valid_models:
            logger.warning(f"No models found with metric '{metric}'")
            return None
        
        # Find the model with the best metric value (higher is better)
        best_model = max(valid_models, key=lambda m: m["metrics"][metric])
        logger.info(f"Best model: {best_model['id']} with {metric} = {best_model['metrics'][metric]}")
        
        return best_model
    
    def load_model(self, model_id=None, model_path=None):
        """
        Load a model from disk by ID or path
        
        Args:
            model_id (str, optional): ID of the model to load
            model_path (str, optional): Direct path to the model file
            
        Returns:
            tensorflow.keras.models.Model: The loaded model
        """
        # Determine the model path
        if model_id is not None:
            # Find the model in the registry
            model_entry = next((m for m in self.model_registry["models"] if m["id"] == model_id), None)
            if model_entry is None:
                logger.error(f"Model with ID {model_id} not found in registry")
                return None
            model_path = model_entry["path"]
        
        if model_path is None:
            logger.error("No models found in registry")
            return None
        
        # Check if the model file exists
        if not os.path.exists(model_path):
            logger.info(f"Model file not found at {model_path}, trying alternatives")
            
            # Try alternative file extensions
            for ext in [".keras", ".h5", ".tf"]:
                alt_path = os.path.splitext(model_path)[0] + ext
                if os.path.exists(alt_path):
                    logger.info(f"Found alternative model file: {alt_path}")
                    model_path = alt_path
                    break
            
            # If still not found, try looking in the models directory
            if not os.path.exists(model_path):
                base_name = os.path.basename(model_path)
                alt_path = os.path.join(self.models_dir, base_name)
                if os.path.exists(alt_path):
                    logger.info(f"Found model in models directory: {alt_path}")
                    model_path = alt_path
                else:
                    # Try finding any model with similar name pattern
                    model_files = [f for f in os.listdir(self.models_dir) 
                                  if f.endswith(('.keras', '.h5', '.tf')) and 'best_model' in f]
                    if model_files:
                        alt_path = os.path.join(self.models_dir, model_files[0])
                        logger.info(f"Found alternative model: {alt_path}")
                        model_path = alt_path
                    else:
                        logger.error(f"Model file not found after attempting format conversion")
                        return None
        
        # Load the model with error handling
        try:
            # Apply monkey patch before loading model
            try:
                from src.utils.monkey_patch import monkeypatch
                monkeypatch()
                logger.info("Applied monkey patch before loading model")
            except ImportError:
                logger.warning("Could not import monkey_patch module")
                
            # First try loading with tf.keras.models.load_model
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Successfully loaded model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            try:
                # Try loading with a custom_objects dictionary for custom layers
                model = tf.keras.models.load_model(model_path, compile=False)
                logger.info(f"Loaded model without compilation from {model_path}")
                # Recompile the model with default settings
                model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                logger.info("Recompiled model with default settings")
                return model
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                return None
    
    def delete_model(self, model_id, delete_files=True):
        """
        Delete a model from the registry and optionally delete the files
        
        Args:
            model_id (str): ID of the model to delete
            delete_files (bool): Whether to delete the model files
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            # Find model in registry
            models = [m for m in self.model_registry["models"] if m["id"] == model_id]
            if not models:
                logger.error(f"Model with ID {model_id} not found in registry")
                return False
            
            model_entry = models[0]
            model_path = model_entry["path"]
            
            # Delete model files if requested
            if delete_files and os.path.exists(model_path):
                os.remove(model_path)
                logger.info(f"Deleted model file: {model_path}")
                
                # Also delete architecture JSON if it exists
                arch_path = model_path.rsplit('.', 1)[0] + "_architecture.json"
                if os.path.exists(arch_path):
                    os.remove(arch_path)
                    logger.info(f"Deleted architecture file: {arch_path}")
            
            # Remove from registry
            self.model_registry["models"] = [m for m in self.model_registry["models"] if m["id"] != model_id]
            
            # Save updated registry
            self._save_registry()
            
            logger.info(f"Model {model_id} deleted from registry")
            return True
        
        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            return False
    
    def save_training_history(self, history, model_id, model_type):
        """
        Save the training history for a model
        
        Args:
            history (tf.keras.callbacks.History): Training history
            model_id (str): ID of the model
            model_type (str): Type of model
            
        Returns:
            str: Path to the saved history file
        """
        try:
            # Convert to JSON-serializable format
            history_dict = {
                'loss': history.history['loss'],
                'accuracy': history.history['accuracy'],
                'val_loss': history.history['val_loss'],
                'val_accuracy': history.history['val_accuracy']
            }
            
            # Create filename
            history_filename = f"{model_type}_training_history.json"
            history_path = os.path.join(self.results_dir, history_filename)
            
            # Save to file
            with open(history_path, 'w') as f:
                json.dump(history_dict, f, indent=4)
            
            logger.info(f"Training history saved to {history_path}")
            
            # Update model entry if it exists
            for model in self.model_registry["models"]:
                if model["id"] == model_id:
                    model["training_history_path"] = history_path
                    self._save_registry()
                    break
            
            return history_path
        
        except Exception as e:
            logger.error(f"Error saving training history: {e}")
            return None
    
    def save_test_data(self, X_test, y_test, model_type):
        """
        Save test data for future evaluation
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            model_type (str): Type of model
        
        Returns:
            tuple: Paths to the saved test data files
        """
        try:
            # Create filenames
            X_test_path = os.path.join(self.results_dir, f"{model_type}_X_test.npy")
            y_test_path = os.path.join(self.results_dir, f"{model_type}_y_test.npy")
            # Save arrays
            np.save(X_test_path, X_test)
            np.save(y_test_path, y_test)
            logger.info(f"Test data saved to {X_test_path} and {y_test_path}")
            return X_test_path, y_test_path
        except Exception as e:
            logger.error(f"Error saving test data: {e}")
            return None, None
            
    def get_latest_model(self, model_type=None):
        """
        Get the most recently created model from the registry

        Args:
            model_type (str, optional): Filter by model type (cnn, mlp, etc)
            
        Returns:
            dict: Latest model entry or None if no models found
        """
        models = self.get_models(model_type)
        if not models:
            return None
            
        # Sort by creation date and return most recent
        return max(models, key=lambda x: x.get("created", ""))
        
    def _scan_for_new_models(self):
        """Scan models directory for unregistered models and add them to registry"""
        registered_paths = [m["path"] for m in self.model_registry["models"]]
        
        # Scan directory for model files
        for file in os.listdir(self.models_dir):
            if file.endswith(".keras") and "_emotion_model" in file:
                model_path = os.path.join(self.models_dir, file)
                
                if model_path not in registered_paths:
                    # Determine model type from filename
                    model_type = file.split("_")[0].lower()
                    
                    # Register model with basic info
                    self.register_model(
                        model_path=model_path,
                        model_type=model_type
                    )
