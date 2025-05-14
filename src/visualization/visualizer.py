"""
visualization.py - Visualization utilities for the speech emotion classification system.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import librosa
import logging

logger = logging.getLogger(__name__)

class EmotionVisualizer:
    """Class for visualizing speech emotion classification results and features."""
    
    def __init__(self, results_dir="results"):
        """Initialize the visualizer with a results directory."""
        self.results_dir = results_dir
        os.makedirs(os.path.join(results_dir, "visualizations"), exist_ok=True)
    
    def plot_waveform(self, audio_data, sr, title="Waveform", save_path=None):
        """Plot audio waveform."""
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(audio_data, sr=sr)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Waveform plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_spectrogram(self, spectrogram, title="Mel Spectrogram", save_path=None):
        """Plot mel spectrogram."""
        plt.figure(figsize=(12, 8))
        librosa.display.specshow(
            spectrogram,
            y_axis='mel',
            x_axis='time',
            cmap='viridis'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Spectrogram plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_mfccs(self, mfccs, title="MFCCs", save_path=None):
        """Plot MFCC features."""
        plt.figure(figsize=(12, 8))
        librosa.display.specshow(
            mfccs,
            x_axis='time',
            cmap='coolwarm'
        )
        plt.colorbar()
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("MFCC Coefficients")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"MFCC plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_training_history(self, history, model_type="model", save_path=None):
        """Plot training history."""
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Model Accuracy', 'Model Loss'))
        
        # Accuracy subplot
        fig.add_trace(
            go.Scatter(y=history['accuracy'], name="Training Accuracy"),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(y=history['val_accuracy'], name="Validation Accuracy"),
            row=1, col=1
        )
        
        # Loss subplot
        fig.add_trace(
            go.Scatter(y=history['loss'], name="Training Loss"),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(y=history['val_loss'], name="Validation Loss"),
            row=1, col=2
        )
        
        fig.update_layout(height=400, width=900, title_text=f"{model_type} Training History")
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Training history plot saved to {save_path}")
        else:
            fig.show()
    
    def plot_confusion_matrix(self, cm, labels=None, title="Confusion Matrix", save_path=None):
        """Plot confusion matrix."""
        if labels is None:
            labels = [str(i) for i in range(len(cm))]
        
        fig = px.imshow(cm,
                       labels=dict(x="Predicted", y="True", color="Count"),
                       x=labels,
                       y=labels,
                       title=title,
                       color_continuous_scale="Blues")
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Confusion matrix plot saved to {save_path}")
        else:
            fig.show()
    
    def plot_feature_distribution(self, features, labels, title="Feature Distribution", save_path=None):
        """Plot distribution of features using dimensionality reduction."""
        # Reduce dimensions using t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features)
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'x': features_2d[:, 0],
            'y': features_2d[:, 1],
            'emotion': [str(l) for l in labels]
        })
        
        # Create scatter plot
        fig = px.scatter(df, x='x', y='y',
                        color='emotion',
                        title=title)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Feature distribution plot saved to {save_path}")
        else:
            fig.show()
    
    def plot_prediction_distribution(self, predictions, true_labels=None, save_path=None):
        """Plot distribution of model predictions."""
        if true_labels is not None:
            df = pd.DataFrame({
                'Predicted': predictions,
                'True': true_labels
            })
            df['Correct'] = df['Predicted'] == df['True']
            
            fig = px.histogram(df, x='Predicted',
                             color='Correct',
                             title='Distribution of Predictions',
                             barmode='group')
        else:
            df = pd.DataFrame({'Predicted': predictions})
            fig = px.histogram(df, x='Predicted',
                             title='Distribution of Predictions')
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Prediction distribution plot saved to {save_path}")
        else:
            fig.show()
    
    def create_report(self, results, model_type="model", save_dir=None):
        """Create a comprehensive visualization report."""
        if save_dir is None:
            save_dir = os.path.join(self.results_dir, "reports")
        os.makedirs(save_dir, exist_ok=True)
        
        # Training history
        if 'history' in results:
            self.plot_training_history(
                results['history'],
                model_type=model_type,
                save_path=os.path.join(save_dir, f"{model_type}_training_history.html")
            )
        
        # Confusion matrix
        if 'confusion_matrix' in results:
            self.plot_confusion_matrix(
                results['confusion_matrix'],
                labels=results.get('labels'),
                title=f"{model_type} Confusion Matrix",
                save_path=os.path.join(save_dir, f"{model_type}_confusion_matrix.html")
            )
        
        # Feature distribution
        if all(k in results for k in ['features', 'labels']):
            self.plot_feature_distribution(
                results['features'],
                results['labels'],
                title=f"{model_type} Feature Distribution",
                save_path=os.path.join(save_dir, f"{model_type}_feature_distribution.html")
            )
        
        # Prediction distribution
        if 'predictions' in results:
            self.plot_prediction_distribution(
                results['predictions'],
                true_labels=results.get('true_labels'),
                save_path=os.path.join(save_dir, f"{model_type}_prediction_distribution.html")
            )
        
        logger.info(f"Visualization report created in {save_dir}")
        return save_dir
