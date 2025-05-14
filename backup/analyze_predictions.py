#!/usr/bin/env python3
# analyze_predictions.py - Script to analyze individual predictions from the model

import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import pandas as pd
import argparse
import traceback

# Import monkey patch first to fix OverflowError
import monkey_patch
monkey_patch.monkeypatch()

import tensorflow as tf
from tensorflow.keras.models import load_model
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data_and_model(model_type="cnn", model_path="models/cnn_emotion_model.keras"):
    """Load test data and the trained model"""
    # Load test data
    X_test = np.load(f"results/{model_type}_X_test.npy")
    y_test = np.load(f"results/{model_type}_y_test.npy")
    logger.info(f"Test data loaded: X_test shape {X_test.shape}, y_test shape {y_test.shape}")
    
    # Load model
    try:
        model = load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        # Try loading with .h5 extension if .keras fails
        if model_path.endswith('.keras'):
            h5_path = model_path.replace('.keras', '.h5')
            logger.info(f"Trying to load model from {h5_path}...")
            model = load_model(h5_path)
            logger.info(f"Model loaded from {h5_path}")
    
    return X_test, y_test, model

def analyze_sample(X_test, y_test, model, sample_index=0):
    """Analyze a specific sample from the test set"""
    # Get the sample
    sample = X_test[sample_index:sample_index+1]
    true_label = y_test[sample_index]
    
    # Get model prediction
    prediction = model.predict(sample)
    predicted_label = np.argmax(prediction[0])
    
    # Map indices to emotion labels (adjust based on your dataset)
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    emotion_labels = emotions[:len(np.unique(y_test))]
    
    logger.info(f"Sample {sample_index}:")
    logger.info(f"True emotion: {emotion_labels[true_label]} (index {true_label})")
    logger.info(f"Predicted emotion: {emotion_labels[predicted_label]} (index {predicted_label})")
    
    # Plot the prediction probabilities
    plt.figure(figsize=(12, 6))
    plt.bar(emotion_labels, prediction[0])
    plt.xlabel('Emotion')
    plt.ylabel('Probability')
    plt.title(f'Prediction Probabilities - Sample {sample_index}')
    plt.ylim(0, 1)  # Set y-axis limit to 0-1 for probability
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs('results/sample_analysis', exist_ok=True)
    
    # Save the figure
    plt.savefig(f'results/sample_analysis/sample_{sample_index}_prediction.png')
    logger.info(f"Prediction plot saved to results/sample_analysis/sample_{sample_index}_prediction.png")
    
    # Plot feature visualization
    visualize_feature_input(sample, sample_index, model_type="cnn")
    
    return true_label, predicted_label, prediction[0]

def visualize_feature_input(sample, sample_index, model_type="cnn"):
    """Visualize the input features for a sample"""
    plt.figure(figsize=(12, 6))
    
    # Reshape sample for visualization based on model type
    if model_type == "cnn":
        # For CNN, we likely have a spectrogram or MFCC features in 2D
        feature_data = sample.squeeze()
        if len(feature_data.shape) > 2:  # If it has a channel dimension
            feature_data = feature_data[:, :, 0]  # Take the first channel
        
        plt.imshow(feature_data, aspect='auto', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Input Features (Spectrogram/MFCC) - Sample {sample_index}')
        plt.xlabel('Time Frames')
        plt.ylabel('Frequency Bins')
    else:
        # For MLP, we likely have 1D features
        plt.plot(sample.squeeze())
        plt.title(f'Input Features - Sample {sample_index}')
        plt.xlabel('Feature Index')
        plt.ylabel('Value')
    
    plt.tight_layout()
    plt.savefig(f'results/sample_analysis/sample_{sample_index}_features.png')
    logger.info(f"Feature visualization saved to results/sample_analysis/sample_{sample_index}_features.png")

def analyze_misclassifications(X_test, y_test, model, top_n=10):
    """Analyze the most confident misclassifications"""
    # Get predictions for all samples
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Find misclassifications
    misclassified_indices = np.where(predicted_labels != y_test)[0]
    
    if len(misclassified_indices) == 0:
        logger.info("No misclassifications found!")
        return
    
    # Get confidence scores for misclassifications
    misclassified_confidences = np.max(predictions[misclassified_indices], axis=1)
    
    # Sort by confidence (highest first)
    sorted_indices = np.argsort(-misclassified_confidences)
    top_misclassified = misclassified_indices[sorted_indices[:top_n]]
    
    # Map indices to emotion labels (adjust based on your dataset)
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    emotion_labels = emotions[:len(np.unique(y_test))]
    
    # Create a dataframe with the results
    results = []
    
    logger.info(f"\nTop {top_n} Most Confident Misclassifications:")
    for i, idx in enumerate(top_misclassified):
        true_label = y_test[idx]
        pred_label = predicted_labels[idx]
        confidence = predictions[idx][pred_label]
        
        logger.info(f"{i+1}. Sample {idx}: True: {emotion_labels[true_label]}, "
                   f"Predicted: {emotion_labels[pred_label]}, Confidence: {confidence:.4f}")
        
        results.append({
            'Sample': idx,
            'True Emotion': emotion_labels[true_label],
            'Predicted Emotion': emotion_labels[pred_label],
            'Confidence': confidence
        })
        
        # Analyze the sample
        analyze_sample(X_test, y_test, model, sample_index=idx)
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv('results/top_misclassifications.csv', index=False)
    logger.info(f"Top misclassifications saved to results/top_misclassifications.csv")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze speech emotion classification model predictions')
    parser.add_argument('--sample', type=int, default=-1, help='Index of specific sample to analyze (-1 to analyze misclassifications)')
    parser.add_argument('--model_type', type=str, default='cnn', help='Model type (cnn or mlp)')
    parser.add_argument('--top_n', type=int, default=5, help='Number of top misclassifications to analyze')
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting prediction analysis")
        
        # Load data and model
        X_test, y_test, model = load_data_and_model(model_type=args.model_type)
        
        if args.sample >= 0 and args.sample < len(X_test):
            # Analyze specific sample
            analyze_sample(X_test, y_test, model, sample_index=args.sample)
        else:
            # Analyze misclassifications
            analyze_misclassifications(X_test, y_test, model, top_n=args.top_n)
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in prediction analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()