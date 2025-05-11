#!/usr/bin/env python3
# model_manager.py - Manage trained models for speech emotion classification

import os
import json
import shutil
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
            model_type (str, optional): Type of models to return
            
        Returns:
            list: List of model entries
        """
        models = self.model_registry["models"]
        
        # Scan models directory to find unregistered models
        self._scan_for_new_models()
        
        # Filter by type if specified
        if model_type:
            models = [m for m in models if m["type"].lower() == model_type.lower()]
        
        # Sort by creation date (newest first)
        models = sorted(models, key=lambda x: x["created"], reverse=True)
        
        return models
    
    def _scan_for_new_models(self):
        """Scan models directory for unregistered models and add them to registry"""
        registered_paths = [m["path"] for m in self.model_registry["models"]]
        
        # Scan directory for model files
        for file in os.listdir(self.models_dir):
            if file.endswith((".keras", ".h5")) and "_emotion_model" in file:
                model_path = os.path.join(self.models_dir, file)
                
                if model_path not in registered_paths:
                    # Determine model type from filename
                    model_type = file.split("_")[0].lower()
                    
                    # Register model with basic info
                    self.register_model(
                        model_path=model_path,
                        model_type=model_type
                    )
    
    def get_model_by_id(self, model_id):
        """
        Get a model entry by its ID
        
        Args:
            model_id (str): ID of the model to retrieve
            
        Returns:
            dict: Model entry or None if not found
        """
        for model in self.model_registry["models"]:
            if model["id"] == model_id:
                return model
        return None
    
    def get_model_by_path(self, model_path):
        """
        Get a model entry by its path
        
        Args:
            model_path (str): Path of the model to retrieve
            
        Returns:
            dict: Model entry or None if not found
        """
        for model in self.model_registry["models"]:
            if model["path"] == model_path:
                return model
        return None
    
    def get_latest_model(self, model_type=None):
        """
        Get the most recently created model of the specified type
        
        Args:
            model_type (str, optional): Type of model to return
            
        Returns:
            dict: Latest model entry or None if no models found
        """
        models = self.get_models(model_type)
        return models[0] if models else None
    
    def load_model(self, model_path=None, model_id=None):
        """
        Load a model from the registry
        
        Args:
            model_path (str, optional): Path to the model file
            model_id (str, optional): ID of the model to load
            
        Returns:
            tf.keras.Model: Loaded model or None if loading fails
        """
        # If neither path nor ID is provided, load the latest model
        if not model_path and not model_id:
            latest_model = self.get_latest_model()
            if latest_model:
                model_path = latest_model["path"]
            else:
                logger.error("No models found in registry")
                return None
        
        # Get model path from ID if provided
        if model_id and not model_path:
            model_entry = self.get_model_by_id(model_id)
            if model_entry:
                model_path = model_entry["path"]
            else:
                logger.error(f"Model with ID {model_id} not found in registry")
                return None
        
        # Load model
        try:
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            
            # Try alternate format (.h5 <-> .keras)
            alt_path = model_path.replace(".keras", ".h5") if model_path.endswith(".keras") else model_path.replace(".h5", ".keras")
            
            if os.path.exists(alt_path):
                try:
                    model = tf.keras.models.load_model(alt_path)
                    logger.info(f"Model loaded from alternate path {alt_path}")
                    return model
                except Exception as e2:
                    logger.error(f"Error loading model from alternate path {alt_path}: {e2}")
            
            return None
    
    def save_model_metrics(self, model_path, metrics):
        """
        Save model metrics to the registry
        
        Args:
            model_path (str): Path to the model file
            metrics (dict): Performance metrics of the model
        """
        model_entry = self.get_model_by_path(model_path)
        
        if model_entry:
            model_entry["metrics"] = metrics
            self._save_registry()
            logger.info(f"Updated metrics for model at {model_path}")
        else:
            # Model not in registry, determine type and register it
            if model_path.endswith("cnn_emotion_model.keras") or model_path.endswith("cnn_emotion_model.h5"):
                model_type = "cnn"
            elif model_path.endswith("mlp_emotion_model.keras") or model_path.endswith("mlp_emotion_model.h5"):
                model_type = "mlp"
            else:
                model_type = "unknown"
            
            self.register_model(model_path, model_type, metrics)
    
    def update_model_description(self, model_id, description):
        """
        Update a model's description in the registry
        
        Args:
            model_id (str): ID of the model to update
            description (str): New description
        """
        model_entry = self.get_model_by_id(model_id)
        
        if model_entry:
            model_entry["description"] = description
            self._save_registry()
            logger.info(f"Updated description for model {model_id}")
            return True
        else:
            logger.error(f"Model with ID {model_id} not found in registry")
            return False
    
    def delete_model(self, model_id, delete_file=False):
        """
        Delete a model from the registry and optionally delete the file
        
        Args:
            model_id (str): ID of the model to delete
            delete_file (bool): Whether to delete the model file
        """
        model_entry = self.get_model_by_id(model_id)
        
        if model_entry:
            # Remove from registry
            self.model_registry["models"] = [m for m in self.model_registry["models"] if m["id"] != model_id]
            self._save_registry()
            
            # Delete file if requested
            if delete_file and os.path.exists(model_entry["path"]):
                try:
                    os.remove(model_entry["path"])
                    logger.info(f"Deleted model file at {model_entry['path']}")
                except Exception as e:
                    logger.error(f"Error deleting model file at {model_entry['path']}: {e}")
            
            logger.info(f"Deleted model {model_id} from registry")
            return True
        else:
            logger.error(f"Model with ID {model_id} not found in registry")
            return False

    def get_model_evaluation_report(self, model_id=None, model_path=None):
        """
        Get the evaluation report for a model
        
        Args:
            model_id (str, optional): ID of the model
            model_path (str, optional): Path to the model file
            
        Returns:
            dict: Evaluation report or None if not found
        """
        # Get model entry
        model_entry = None
        if model_id:
            model_entry = self.get_model_by_id(model_id)
        elif model_path:
            model_entry = self.get_model_by_path(model_path)
        
        if not model_entry:
            logger.error(f"Model not found in registry")
            return None
        
        # Check for model evaluation report
        model_type = model_entry["type"]
        report_path = os.path.join(self.results_dir, "model_evaluation_report.json")
        
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r') as f:
                    report = json.load(f)
                logger.info(f"Loaded evaluation report for {model_type} model")
                return report
            except Exception as e:
                logger.error(f"Error loading evaluation report: {e}")
        
        # If no general report, try model-specific report
        model_report_path = os.path.join(self.results_dir, f"{model_type}_evaluation_report.json")
        if os.path.exists(model_report_path):
            try:
                with open(model_report_path, 'r') as f:
                    report = json.load(f)
                logger.info(f"Loaded model-specific evaluation report for {model_type} model")
                return report
            except Exception as e:
                logger.error(f"Error loading model-specific evaluation report: {e}")
        
        # Check metrics in model entry
        if model_entry.get("metrics"):
            logger.info(f"Using metrics from model registry for {model_type} model")
            return model_entry["metrics"]
        
        return None

    def generate_model_comparison_chart(self, model_ids=None, metric='accuracy'):
        """
        Generate a comparison chart of multiple models
        
        Args:
            model_ids (list, optional): List of model IDs to compare
            metric (str, optional): Metric to compare (accuracy, precision, recall, f1)
            
        Returns:
            dict: Chart data in a format suitable for visualization
        """
        # If no model IDs provided, use all models
        if not model_ids:
            models = self.get_models()
            model_ids = [m['id'] for m in models]
        
        # Get model entries
        model_entries = []
        for model_id in model_ids:
            model_entry = self.get_model_by_id(model_id)
            if model_entry and model_entry.get('metrics', {}).get(metric) is not None:
                model_entries.append(model_entry)
        
        if not model_entries:
            logger.error(f"No models found with metric {metric}")
            return None
        
        # Prepare chart data
        chart_data = {
            'labels': [m['id'] for m in model_entries],
            'datasets': [{
                'label': metric.capitalize(),
                'data': [m['metrics'].get(metric, 0) for m in model_entries],
                'backgroundColor': ['rgba(54, 162, 235, 0.2)'] * len(model_entries),
                'borderColor': ['rgba(54, 162, 235, 1)'] * len(model_entries),
                'borderWidth': 1
            }]
        }
        
        # Add model types as additional info
        chart_data['model_types'] = [m['type'] for m in model_entries]
        chart_data['created_dates'] = [m['created'] for m in model_entries]
        
        return chart_data
    
    def generate_model_metrics_radar_chart(self, model_id):
        """
        Generate a radar chart of a model's metrics
        
        Args:
            model_id (str): ID of the model
            
        Returns:
            dict: Chart data in a format suitable for visualization
        """
        model_entry = self.get_model_by_id(model_id)
        
        if not model_entry or not model_entry.get('metrics'):
            logger.error(f"Model {model_id} not found or has no metrics")
            return None
        
        metrics = model_entry['metrics']
        
        # Prepare chart data
        chart_data = {
            'labels': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'datasets': [{
                'label': f"{model_entry['type'].upper()} Model",
                'data': [
                    metrics.get('accuracy', 0), 
                    metrics.get('precision', 0), 
                    metrics.get('recall', 0), 
                    metrics.get('f1', 0)
                ],
                'backgroundColor': 'rgba(255, 99, 132, 0.2)',
                'borderColor': 'rgba(255, 99, 132, 1)',
                'borderWidth': 1
            }]
        }
        
        return chart_data

    def compare_models(self, model_ids):
        """
        Compare multiple models side by side
        
        Args:
            model_ids (list): List of model IDs to compare
            
        Returns:
            dict: Comparison data for the models
        """
        comparison = {
            'models': [],
            'metrics': {},
            'additional_info': {}
        }
        
        for model_id in model_ids:
            model_entry = self.get_model_by_id(model_id)
            if model_entry:
                # Add to models list
                comparison['models'].append({
                    'id': model_id,
                    'type': model_entry['type'],
                    'created': model_entry['created'],
                    'description': model_entry.get('description', ''),
                    'size_mb': model_entry.get('size_mb', 0)
                })
                
                # Add metrics
                metrics = model_entry.get('metrics', {})
                for metric_name, metric_value in metrics.items():
                    if metric_name not in comparison['metrics']:
                        comparison['metrics'][metric_name] = []
                    
                    # Add the metric value for this model
                    comparison['metrics'][metric_name].append(metric_value)
                
                # Add additional info
                for key, value in model_entry.items():
                    if key not in ['id', 'type', 'created', 'description', 'size_mb', 'metrics', 'path']:
                        if key not in comparison['additional_info']:
                            comparison['additional_info'][key] = []
                        comparison['additional_info'][key].append(value)
        
        return comparison


if __name__ == "__main__":
    # Example usage
    manager = ModelManager()
    
    # Get all registered models
    models = manager.get_models()
    print(f"Found {len(models)} registered models")
    
    for model in models:
        print(f"Model ID: {model['id']}")
        print(f"  Path: {model['path']}")
        print(f"  Type: {model['type']}")
        print(f"  Created: {model['created']}")
        print(f"  Size: {model.get('size_mb', 'Unknown')} MB")
        print(f"  Metrics: {model.get('metrics', {})}")
        print()
    
    # Load the latest model
    latest_model = manager.get_latest_model()
    if (latest_model):
        print(f"Latest model: {latest_model['id']}")
        model = manager.load_model(model_id=latest_model['id'])
        if model:
            print("Model loaded successfully")
            model.summary()