import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import librosa
import librosa.display
import argparse
import logging
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResultsVisualizer:
    """
    Class for visualizing speech emotion classification results.
    """
    def __init__(self, model_path, results_dir='results', interactive=True):
        """
        Initialize the ResultsVisualizer.
        
        Args:
            model_path (str): Path to the trained model.
            results_dir (str): Directory to save visualization results.
            interactive (bool): Whether to create interactive visualizations.
        """
        self.model_path = model_path
        self.results_dir = results_dir
        self.interactive = interactive
        self.model = None
        self.emotion_labels = ['calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        
        # Create visualization directories
        os.makedirs(os.path.join(results_dir, 'visualizations'), exist_ok=True)
        
        # Load model
        self._load_model()
        
    def _load_model(self):
        """
        Load the trained model.
        """
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def visualize_feature_importance(self, X_test, y_test, feature_names=None):
        """
        Visualize feature importance using permutation importance.
        
        Args:
            X_test (numpy.ndarray): Test features.
            y_test (numpy.ndarray): Test labels.
            feature_names (list): Names of features.
        """
        try:
            # Only applicable for MLP models with 1D features
            if len(X_test.shape) > 2:
                logger.info("Feature importance visualization not applicable for CNN models")
                return
            
            from sklearn.inspection import permutation_importance
            
            # Create a function to convert keras model to sklearn compatible
            def model_predict(X):
                y_pred = self.model.predict(X)
                return np.argmax(y_pred, axis=1)
            
            # Calculate permutation importance
            r = permutation_importance(
                model_predict, X_test, y_test, 
                n_repeats=10, random_state=42, n_jobs=-1
            )
            
            # Sort importance
            importance = r.importances_mean
            indices = np.argsort(importance)[::-1]
            
            if feature_names is None:
                feature_names = [f"Feature {i}" for i in range(X_test.shape[1])]
                
            # Create bar plot
            plt.figure(figsize=(12, 8))
            plt.title("Feature Importance (Permutation)")
            plt.bar(range(X_test.shape[1]), importance[indices], align='center')
            plt.xticks(range(X_test.shape[1]), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'visualizations', 'feature_importance.png'), dpi=300)
            
            if self.interactive:
                # Create interactive bar plot with plotly
                fig = px.bar(
                    x=[feature_names[i] for i in indices],
                    y=importance[indices],
                    title="Feature Importance (Permutation)",
                    labels={'x': 'Feature', 'y': 'Importance'},
                    color=importance[indices],
                    color_continuous_scale='Viridis',
                )
                fig.update_layout(xaxis_tickangle=-45)
                fig.write_html(os.path.join(self.results_dir, 'visualizations', 'feature_importance.html'))
            
            logger.info(f"Feature importance visualization saved to {self.results_dir}/visualizations/")
            
        except Exception as e:
            logger.error(f"Error visualizing feature importance: {e}")
    
    def visualize_tsne(self, X, y, perplexity=30, n_iter=1000):
        """
        Visualize data distribution using t-SNE.
        
        Args:
            X (numpy.ndarray): Features.
            y (numpy.ndarray): Labels.
            perplexity (int): Perplexity parameter for t-SNE.
            n_iter (int): Number of iterations for t-SNE.
        """
        try:
            # Reshape X if it's for CNN
            if len(X.shape) > 2:
                X_flat = X.reshape(X.shape[0], -1)
            else:
                X_flat = X
            
            # Apply t-SNE
            tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
            X_tsne = tsne.fit_transform(X_flat)
            
            # Create DataFrame for easier plotting
            df = pd.DataFrame({
                'x': X_tsne[:, 0],
                'y': X_tsne[:, 1],
                'emotion': [self.emotion_labels[i] for i in y]
            })
            
            # Create scatter plot
            plt.figure(figsize=(12, 10))
            sns.scatterplot(data=df, x='x', y='y', hue='emotion', palette='viridis')
            plt.title('t-SNE Visualization of Feature Space')
            plt.savefig(os.path.join(self.results_dir, 'visualizations', 'tsne_visualization.png'), dpi=300)
            
            if self.interactive:
                # Create interactive plot with plotly
                fig = px.scatter(
                    df, x='x', y='y', color='emotion',
                    title='t-SNE Visualization of Feature Space',
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                fig.update_traces(marker=dict(size=10))
                fig.write_html(os.path.join(self.results_dir, 'visualizations', 'tsne_visualization.html'))
            
            logger.info(f"t-SNE visualization saved to {self.results_dir}/visualizations/")
            
        except Exception as e:
            logger.error(f"Error visualizing t-SNE: {e}")
    
    def visualize_confusion_matrix(self, y_true, y_pred):
        """
        Create an enhanced confusion matrix visualization.
        
        Args:
            y_true (numpy.ndarray): True labels.
            y_pred (numpy.ndarray): Predicted labels.
        """
        try:
            from sklearn.metrics import confusion_matrix
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Create heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                cm_normalized, annot=cm, fmt='d',
                cmap='Blues', xticklabels=self.emotion_labels,
                yticklabels=self.emotion_labels
            )
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'visualizations', 'enhanced_confusion_matrix.png'), dpi=300)
            
            if self.interactive:
                # Create interactive heatmap with plotly
                fig = go.Figure(data=go.Heatmap(
                    z=cm_normalized,
                    x=self.emotion_labels,
                    y=self.emotion_labels,
                    text=cm,
                    texttemplate="%{text}",
                    colorscale='Blues'
                ))
                
                fig.update_layout(
                    title='Confusion Matrix',
                    xaxis=dict(title='Predicted Label'),
                    yaxis=dict(title='True Label')
                )
                
                fig.write_html(os.path.join(self.results_dir, 'visualizations', 'enhanced_confusion_matrix.html'))
            
            logger.info(f"Enhanced confusion matrix saved to {self.results_dir}/visualizations/")
            
        except Exception as e:
            logger.error(f"Error visualizing confusion matrix: {e}")
    
    def visualize_spectrogram(self, audio_path, emotion, prediction=None):
        """
        Visualize spectrogram of an audio file.
        
        Args:
            audio_path (str): Path to the audio file.
            emotion (str): True emotion.
            prediction (str): Predicted emotion.
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=None)
            
            # Create spectrogram
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            
            # Create mel spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram')
            
            plt.subplot(2, 1, 2)
            librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            
            if prediction is not None:
                plt.title(f'Mel Spectrogram (True: {emotion}, Predicted: {prediction})')
            else:
                plt.title(f'Mel Spectrogram (Emotion: {emotion})')
                
            plt.tight_layout()
            
            # Extract filename for saving
            filename = os.path.basename(audio_path).split('.')[0]
            plt.savefig(os.path.join(self.results_dir, 'visualizations', f'{filename}_spectrogram.png'), dpi=300)
            
            logger.info(f"Spectrogram visualization saved to {self.results_dir}/visualizations/")
            
        except Exception as e:
            logger.error(f"Error visualizing spectrogram: {e}")
    
    def visualize_model_architecture(self):
        """
        Visualize the model architecture.
        """
        try:
            # Create plot of model architecture
            plot_path = os.path.join(self.results_dir, 'visualizations', 'model_architecture.png')
            tf.keras.utils.plot_model(
                self.model, 
                to_file=plot_path,
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                dpi=96
            )
            
            # Create a summary text file
            summary_path = os.path.join(self.results_dir, 'visualizations', 'model_summary.txt')
            
            # We need to redirect the summary to a file
            from contextlib import redirect_stdout
            with open(summary_path, 'w') as f:
                with redirect_stdout(f):
                    self.model.summary()
            
            logger.info(f"Model architecture visualization saved to {plot_path}")
            logger.info(f"Model summary saved to {summary_path}")
            
        except Exception as e:
            logger.error(f"Error visualizing model architecture: {e}")
            
    def visualize_history(self, history_path):
        """
        Visualize training history more extensively.
        
        Args:
            history_path (str): Path to the training history JSON file.
        """
        try:
            # Load training history
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            # Create subplots
            if self.interactive:
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Model Accuracy', 'Model Loss'),
                    shared_xaxes=True,
                    vertical_spacing=0.1
                )
                
                # Add accuracy traces
                fig.add_trace(
                    go.Scatter(x=list(range(1, len(history['accuracy'])+1)), y=history['accuracy'],
                               mode='lines+markers', name='Training Accuracy'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=list(range(1, len(history['val_accuracy'])+1)), y=history['val_accuracy'],
                               mode='lines+markers', name='Validation Accuracy'),
                    row=1, col=1
                )
                
                # Add loss traces
                fig.add_trace(
                    go.Scatter(x=list(range(1, len(history['loss'])+1)), y=history['loss'],
                               mode='lines+markers', name='Training Loss'),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=list(range(1, len(history['val_loss'])+1)), y=history['val_loss'],
                               mode='lines+markers', name='Validation Loss'),
                    row=2, col=1
                )
                
                # Update layout
                fig.update_layout(
                    title='Training History',
                    xaxis_title='Epoch',
                    yaxis_title='Metric Value',
                    height=800
                )
                
                # Save interactive plot
                fig.write_html(os.path.join(self.results_dir, 'visualizations', 'training_history_interactive.html'))
            
            # Create static plots
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 1, 1)
            plt.plot(history['accuracy'], label='Training Accuracy')
            plt.plot(history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 1, 2)
            plt.plot(history['loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'visualizations', 'training_history_detailed.png'), dpi=300)
            
            logger.info(f"Training history visualization saved to {self.results_dir}/visualizations/")
            
        except Exception as e:
            logger.error(f"Error visualizing training history: {e}")
            
    def analyze_misclassifications(self, X_test, y_test, audio_paths=None):
        """
        Analyze misclassified examples.
        
        Args:
            X_test (numpy.ndarray): Test features.
            y_test (numpy.ndarray): Test labels.
            audio_paths (list): List of audio file paths corresponding to test samples.
        """
        try:
            # Get predictions
            y_pred = self.model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            # Find misclassified examples
            misclassified_indices = np.where(y_pred_classes != y_test)[0]
            
            # Calculate confidence scores
            confidence_scores = np.max(y_pred, axis=1)
            
            # Create a DataFrame with misclassifications
            misclassified_data = []
            
            for idx in misclassified_indices:
                entry = {
                    'index': idx,
                    'true_label': self.emotion_labels[y_test[idx]],
                    'predicted_label': self.emotion_labels[y_pred_classes[idx]],
                    'confidence': confidence_scores[idx],
                    'audio_path': audio_paths[idx] if audio_paths is not None else None
                }
                misclassified_data.append(entry)
                
            df_misclassified = pd.DataFrame(misclassified_data)
            df_misclassified.sort_values('confidence', ascending=False, inplace=True)
            
            # Save misclassifications to CSV
            csv_path = os.path.join(self.results_dir, 'visualizations', 'misclassifications.csv')
            df_misclassified.to_csv(csv_path, index=False)
            
            # Create a bar chart showing the frequency of misclassification types
            misclass_pairs = df_misclassified.apply(
                lambda row: f"{row['true_label']} → {row['predicted_label']}", axis=1
            )
            
            plt.figure(figsize=(14, 8))
            counts = misclass_pairs.value_counts().sort_values(ascending=False).head(15)
            counts.plot(kind='barh')
            plt.title('Top 15 Misclassification Types')
            plt.xlabel('Count')
            plt.ylabel('True → Predicted')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'visualizations', 'misclassification_types.png'), dpi=300)
            
            if self.interactive:
                # Create interactive visualization
                fig = px.bar(
                    x=counts.values,
                    y=counts.index,
                    title='Misclassification Types',
                    labels={'x': 'Count', 'y': 'True → Predicted'},
                    orientation='h',
                    color=counts.values,
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                fig.write_html(os.path.join(self.results_dir, 'visualizations', 'misclassification_types.html'))
            
            logger.info(f"Misclassification analysis saved to {self.results_dir}/visualizations/")
            
            return df_misclassified
            
        except Exception as e:
            logger.error(f"Error analyzing misclassifications: {e}")
            
    def generate_report(self, test_metrics, misclassified_df=None):
        """
        Generate an HTML report summarizing all the results.
        
        Args:
            test_metrics (dict): Dictionary containing test metrics.
            misclassified_df (DataFrame): DataFrame with misclassified examples.
        """
        try:
            # Create a simple HTML report
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Speech Emotion Classification Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    h1 { color: #333; }
                    h2 { color: #666; margin-top: 30px; }
                    table { border-collapse: collapse; width: 100%; margin-top: 20px; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    tr:nth-child(even) { background-color: #f9f9f9; }
                    img { max-width: 100%; margin-top: 20px; }
                    .metrics { display: flex; flex-wrap: wrap; }
                    .metric-card { 
                        background-color: #f0f8ff; 
                        border-radius: 5px; 
                        padding: 15px; 
                        margin: 10px; 
                        flex: 0 0 200px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                    .metric-value { font-size: 24px; font-weight: bold; margin-top: 10px; }
                </style>
            </head>
            <body>
                <h1>Speech Emotion Classification Report</h1>
                
                <h2>Performance Metrics</h2>
                <div class="metrics">
            """
            
            # Add performance metrics
            metrics_to_show = [
                ('accuracy', 'Accuracy'),
                ('precision_avg', 'Precision'),
                ('recall_avg', 'Recall'),
                ('f1_avg', 'F1 Score')
            ]
            
            for metric_key, metric_name in metrics_to_show:
                if metric_key in test_metrics:
                    value = test_metrics[metric_key]
                    html_content += f"""
                    <div class="metric-card">
                        <div>{metric_name}</div>
                        <div class="metric-value">{value:.4f}</div>
                    </div>
                    """
            
            html_content += """
                </div>
                
                <h2>Visualizations</h2>
            """
            
            # Add visualization images
            visualization_files = [
                ('confusion_matrix.png', 'Confusion Matrix'),
                ('enhanced_confusion_matrix.png', 'Enhanced Confusion Matrix'),
                ('training_history_detailed.png', 'Training History'),
                ('tsne_visualization.png', 't-SNE Visualization'),
                ('misclassification_types.png', 'Misclassification Types')
            ]
            
            for file_name, title in visualization_files:
                file_path = os.path.join('visualizations', file_name)
                if os.path.exists(os.path.join(self.results_dir, file_path)):
                    html_content += f"""
                    <h3>{title}</h3>
                    <img src="{file_path}" alt="{title}">
                    """
            
            # Add interactive visualization links if available
            if self.interactive:
                html_content += """
                <h2>Interactive Visualizations</h2>
                <ul>
                """
                
                interactive_files = [
                    ('enhanced_confusion_matrix.html', 'Interactive Confusion Matrix'),
                    ('training_history_interactive.html', 'Interactive Training History'),
                    ('tsne_visualization.html', 'Interactive t-SNE Visualization'),
                    ('misclassification_types.html', 'Interactive Misclassification Analysis')
                ]
                
                for file_name, title in interactive_files:
                    file_path = os.path.join('visualizations', file_name)
                    if os.path.exists(os.path.join(self.results_dir, file_path)):
                        html_content += f"""
                        <li><a href="{file_path}" target="_blank">{title}</a></li>
                        """
                
                html_content += """
                </ul>
                """
            
            # Add misclassification analysis if available
            if misclassified_df is not None and not misclassified_df.empty:
                html_content += """
                <h2>Top Misclassifications</h2>
                <table>
                    <tr>
                        <th>True Emotion</th>
                        <th>Predicted Emotion</th>
                        <th>Confidence</th>
                    </tr>
                """
                
                for _, row in misclassified_df.head(10).iterrows():
                    html_content += f"""
                    <tr>
                        <td>{row['true_label']}</td>
                        <td>{row['predicted_label']}</td>
                        <td>{row['confidence']:.4f}</td>
                    </tr>
                    """
                
                html_content += """
                </table>
                """
            
            # Close HTML document
            html_content += """
            </body>
            </html>
            """
            
            # Save HTML report
            report_path = os.path.join(self.results_dir, 'emotion_classification_report.html')
            with open(report_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Report generated and saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Visualize speech emotion classification results')
    parser.add_argument('--model_path', required=True, help='Path to the trained model')
    parser.add_argument('--results_dir', default='results', help='Directory to save visualization results')
    parser.add_argument('--interactive', action='store_true', help='Create interactive visualizations')
    parser.add_argument('--test_data', help='Path to the test data features (numpy file)')
    parser.add_argument('--test_labels', help='Path to the test data labels (numpy file)')
    parser.add_argument('--history', help='Path to the training history JSON file')
    parser.add_argument('--audio_dir', help='Directory with audio files for spectrogram visualization')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Create visualizer
    visualizer = ResultsVisualizer(args.model_path, args.results_dir, args.interactive)
    
    # Visualize model architecture
    visualizer.visualize_model_architecture()
    
    # Load test data if provided
    if args.test_data and args.test_labels:
        X_test = np.load(args.test_data)
        y_test = np.load(args.test_labels)
        
        # Get predictions
        y_pred = visualizer.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Visualize confusion matrix
        visualizer.visualize_confusion_matrix(y_test, y_pred_classes)
        
        # Visualize t-SNE
        visualizer.visualize_tsne(X_test, y_test)
        
        # Analyze misclassifications
        misclassified_df = visualizer.analyze_misclassifications(X_test, y_test)
        
        # If it's a 1D feature model (MLP), visualize feature importance
        if len(X_test.shape) == 2:
            visualizer.visualize_feature_importance(X_test, y_test)
    
    # Visualize training history if provided
    if args.history:
        visualizer.visualize_history(args.history)
    
    # Visualize spectrograms of some audio files if provided
    if args.audio_dir:
        audio_files = [os.path.join(args.audio_dir, f) for f in os.listdir(args.audio_dir) 
                     if f.endswith('.wav')][:5]  # Just visualize a few examples
        
        for audio_path in audio_files:
            # Extract emotion from filename (assuming format contains emotion name)
            filename = os.path.basename(audio_path)
            for emotion in visualizer.emotion_labels:
                if emotion in filename.lower():
                    visualizer.visualize_spectrogram(audio_path, emotion)
                    break
    
    # Generate report
    if args.test_data and args.test_labels:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_classes, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'precision_avg': precision,
            'recall_avg': recall,
            'f1_avg': f1
        }
        
        visualizer.generate_report(metrics, misclassified_df)
    else:
        # Generate a report with just the visualizations
        visualizer.generate_report({})