#!/usr/bin/env python3
# run_model.py - Script to load and run the saved emotion classification model

import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
import pandas as pd

# Import monkey patch first to fix OverflowError
import monkey_patch
monkey_patch.monkeypatch()

import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_test_data(model_type="cnn"):
    """Load the test data saved during model training"""
    try:
        X_test = np.load(f"results/{model_type}_X_test.npy")
        y_test = np.load(f"results/{model_type}_y_test.npy")
        logger.info(f"Test data loaded: X_test shape {X_test.shape}, y_test shape {y_test.shape}")
        return X_test, y_test
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise

def load_saved_model(model_path="models/cnn_emotion_model.keras"):
    """Load the saved Keras model"""
    try:
        model = load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Model summary:")
        model.summary(print_fn=logger.info)
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # Try loading with .h5 extension if .keras fails
        if model_path.endswith('.keras'):
            try:
                h5_path = model_path.replace('.keras', '.h5')
                logger.info(f"Trying to load model from {h5_path}...")
                model = load_model(h5_path)
                logger.info(f"Model loaded from {h5_path}")
                return model
            except Exception as e2:
                logger.error(f"Error loading model from H5: {e2}")
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data"""
    try:
        # Model evaluation
        loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
        logger.info(f"Test loss: {loss:.4f}")
        logger.info(f"Test accuracy: {accuracy:.4f}")
        
        # Get predictions
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Get classification metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return y_pred, y_pred_probs, report, cm
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise

def plot_results(y_test, y_pred, cm):
    """Plot confusion matrix and other visualizations"""
    try:
        # Map emotion indices to labels (adjust these based on your dataset)
        emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        emotion_labels = [emotions[i] for i in range(len(np.unique(y_test)))]
        
        # Create confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=emotion_labels, 
                   yticklabels=emotion_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('results/model_evaluation_cm.png')
        logger.info(f"Confusion matrix plot saved to results/model_evaluation_cm.png")
        
        # Plot prediction distribution
        plt.figure(figsize=(10, 6))
        pd.Series(y_pred).value_counts().sort_index().plot(kind='bar')
        plt.xticks(range(len(emotion_labels)), emotion_labels, rotation=45)
        plt.xlabel('Emotion')
        plt.ylabel('Count')
        plt.title('Distribution of Predicted Emotions')
        plt.tight_layout()
        plt.savefig('results/prediction_distribution.png')
        logger.info(f"Prediction distribution plot saved to results/prediction_distribution.png")
        
    except Exception as e:
        logger.error(f"Error plotting results: {e}")
        raise

def main():
    """Main function to run the model evaluation"""
    try:
        logger.info("Starting model evaluation")
        
        # Load test data
        X_test, y_test = load_test_data(model_type="cnn")
        
        # Load model
        model = load_saved_model()
        
        # Evaluate model
        y_pred, y_pred_probs, report, cm = evaluate_model(model, X_test, y_test)
        
        # Plot results
        plot_results(y_test, y_pred, cm)
        
        # Save evaluation results
        with open('results/model_evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=4)
        logger.info(f"Evaluation report saved to results/model_evaluation_report.json")
        
        logger.info("Model evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")

if __name__ == "__main__":
    main()