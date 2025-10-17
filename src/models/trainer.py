import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                           confusion_matrix, classification_report, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

logger = logging.getLogger(__name__)

# Monkey patch is now applied in src/__init__.py

class ModelTrainer:
    """
    Class for training and evaluating speech emotion classification models.
    """
    def __init__(self, model, model_type='cnn'):
        """Initialize ModelTrainer."""
        self.model = model
        self.model_type = model_type
        self.history = None
        self.training_time = None
        
        # Create required directories using config
        try:
            from src.core import config
            cfg = config.Config()
            results_dir = cfg.paths.results_dir
            reports_dir = os.path.join(results_dir, 'reports')
            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(reports_dir, exist_ok=True)
        except ImportError:
            # Fallback to hardcoded paths
            for dir_path in ['results', 'results/reports']:
                os.makedirs(dir_path, exist_ok=True)
            
    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=50, callbacks=None):
        """Train the model with proper error handling and advanced training features."""
        try:
            start_time = time.time()
            logger.info(f"Starting {self.model_type.upper()} model training for {epochs} epochs with batch size {batch_size}")
            
            # Setup default callbacks if none provided
            if callbacks is None:
                # Get models directory from config
                try:
                    from src.core import config
                    cfg = config.Config()
                    models_dir = cfg.paths.models_dir
                except ImportError:
                    models_dir = 'models'

                # Create models directory if it doesn't exist
                os.makedirs(models_dir, exist_ok=True)

                # Setup model checkpoint to save best model
                checkpoint_path = os.path.join(models_dir, f'{self.model_type}_best_model.keras')
                checkpoint = tf.keras.callbacks.ModelCheckpoint(
                    checkpoint_path,
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min',
                    verbose=1
                )
                
                # Setup early stopping with reasonable patience
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,  # Increased patience for better convergence
                    restore_best_weights=True,
                    verbose=1
                )
                
                # Add ReduceLROnPlateau for adaptive learning rate
                reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,  # Reduce learning rate by half
                    patience=5,  # Wait 5 epochs before reducing
                    min_lr=1e-6,
                    verbose=1
                )
                
                # Add TensorBoard logging
                try:
                    from src.core import config
                    cfg = config.Config()
                    logs_dir = cfg.paths.logs_dir
                except ImportError:
                    logs_dir = 'logs'
                log_dir = os.path.join(logs_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
                tensorboard = tf.keras.callbacks.TensorBoard(
                    log_dir=log_dir,
                    histogram_freq=1,
                    update_freq='epoch',
                    profile_batch=0  # Disable profiling for better performance
                )
                
                # Setup callbacks list
                callbacks = [checkpoint, early_stopping, reduce_lr, tensorboard]
                logger.info("Using enhanced callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, and TensorBoard")
            
            # Data preprocessing and validation
            # Convert input data to proper dtype if needed
            X_train = np.asarray(X_train, dtype=np.float32)
            y_train = np.asarray(y_train, dtype=np.int32)
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.int32)
            
            # Log data shapes
            logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            logger.info(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
            
            # Check for NaN values
            if np.isnan(X_train).any() or np.isnan(X_val).any():
                logger.warning("NaN values detected in training data. Replacing with zeros.")
                X_train = np.nan_to_num(X_train)
                X_val = np.nan_to_num(X_val)
            
            # Check for class imbalance and calculate class weights
            class_counts = np.bincount(y_train)
            logger.info(f"Class distribution: {class_counts}")
            
            # Always use class weights for better training
            from sklearn.utils.class_weight import compute_class_weight
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
            logger.info(f"Using class weights: {class_weight_dict}")
            
            # Check for severe imbalance
            if len(class_counts) > 0 and (max(class_counts) / min(class_counts) > 3):
                logger.warning(f"Severe class imbalance detected. Max/min ratio: {max(class_counts) / min(class_counts):.2f}")
                logger.info("Using class weights to address imbalance")
            
            # Configure mixed precision if available (for faster training on compatible GPUs)
            try:
                mixed_precision = False
                if tf.config.list_physical_devices('GPU'):
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    mixed_precision = True
                    logger.info("Using mixed precision training for performance improvement")
            except Exception as mp_error:
                logger.info(f"Mixed precision not available or not supported: {mp_error}")
            
            # Train the model with class weights
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                class_weight=class_weight_dict,  # Apply class weights
                verbose=1
            )
            
            self.training_time = time.time() - start_time
            logger.info(f"Training completed in {self.training_time:.2f} seconds ({self.training_time/60:.2f} minutes)")
            
            # Plot training history
            self._plot_training_history()
            return self.history
        except Exception as e:
            logger.error(f"Error during training: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def evaluate(self, X_test, y_test, emotion_labels=None):
        """Evaluate model with comprehensive metrics and visualizations."""
        try:
            # Basic evaluation
            metrics = {}
            loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
            logger.info(f"Test loss: {loss:.4f}")
            logger.info(f"Test accuracy: {accuracy:.4f}")
            
            # Get predictions
            y_pred = self.model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            # Generate classification report
            report = classification_report(y_test, y_pred_classes, output_dict=True)
            cm = confusion_matrix(y_test, y_pred_classes)
            
            # Calculate additional metrics
            from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
            
            # Convert to one-hot encoding for ROC AUC calculation
            try:
                from sklearn.preprocessing import OneHotEncoder
                encoder = OneHotEncoder(sparse_output=False)
                y_test_onehot = encoder.fit_transform(y_test.reshape(-1, 1))
                
                # Calculate ROC AUC for multi-class (one-vs-rest)
                roc_auc = roc_auc_score(y_test_onehot, y_pred, multi_class='ovr', average='weighted')
                logger.info(f"ROC AUC Score (weighted): {roc_auc:.4f}")
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
                roc_auc = None
            
            # Store all metrics
            metrics = {
                'accuracy': float(accuracy),
                'loss': float(loss),
                'precision_avg': float(report['weighted avg']['precision']),
                'recall_avg': float(report['weighted avg']['recall']),
                'f1_avg': float(report['weighted avg']['f1-score']),
                'roc_auc': float(roc_auc) if roc_auc is not None else None,
                'per_class_metrics': report,
                'confusion_matrix': cm.tolist()
            }
            
            # Generate visualizations
            self._plot_confusion_matrix(y_test, y_pred_classes, emotion_labels)
            self._plot_prediction_distribution(y_test, y_pred_classes, emotion_labels)
            
            # Save metrics to CSV if emotion labels are provided
            if emotion_labels:
                self._save_metrics_to_csv(metrics, emotion_labels)
                
                # Generate per-class metrics visualization
                self._plot_per_class_metrics(report, emotion_labels)
            
            return metrics
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'accuracy': 0.0, 'loss': float('inf'), 'error': str(e)}

    def save_model(self, filepath):
        """Save model with proper error handling."""
        try:
            filepath = str(filepath)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            base_path = filepath.rsplit('.', 1)[0] if '.' in filepath else filepath
            keras_path = f"{base_path}.keras"

            # Primary save (Keras 2/3 compatible by extension)
            self.model.save(keras_path)
            logger.info(f"Model saved to {keras_path}")

            # Optional H5 for backward compatibility (best-effort)
            h5_path = f"{base_path}.h5"
            try:
                self.model.save(h5_path)
                logger.info(f"Model also saved in h5 format at {h5_path}")
            except Exception as h5_err:
                logger.warning(f"Could not save in h5 format: {h5_err}")

            # Save architecture JSON
            arch_path = f"{base_path}_architecture.json"
            with open(arch_path, 'w') as f:
                f.write(self.model.to_json())
            logger.info(f"Model architecture saved to {arch_path}")

            return keras_path
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def _save_metrics_to_csv(self, metrics, emotion_labels):
        """Save metrics with proper error handling."""
        try:
            if 'per_class_metrics' not in metrics:
                return
                
            os.makedirs('results/reports', exist_ok=True)
            
            class_metrics = []
            for i, label in enumerate(emotion_labels):
                if str(i) in metrics['per_class_metrics']:
                    class_metrics.append({
                        'emotion': label,
                        'precision': metrics['per_class_metrics'][str(i)]['precision'],
                        'recall': metrics['per_class_metrics'][str(i)]['recall'],
                        'f1_score': metrics['per_class_metrics'][str(i)]['f1-score'],
                        'support': metrics['per_class_metrics'][str(i)]['support']
                    })
            
            df = pd.DataFrame(class_metrics)
            csv_path = f'results/reports/{self.model_type}_class_metrics.csv'
            df.to_csv(csv_path, index=False)
            logger.info(f"Class metrics saved to {csv_path}")
            
        except Exception as e:
            logger.error(f"Error saving metrics to CSV: {e}")
            
    def _plot_confusion_matrix(self, y_true, y_pred, emotion_labels=None):
        """Plot confusion matrix with proper error handling."""
        try:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 8))

            # Handle label mismatch - use only labels that exist in the data
            if emotion_labels:
                n_classes = cm.shape[0]
                if len(emotion_labels) > n_classes:
                    # Use subset of labels that match the confusion matrix size
                    display_labels = emotion_labels[:n_classes]
                    logger.warning(f"Using subset of emotion labels ({n_classes}/{len(emotion_labels)}) to match confusion matrix size")
                else:
                    display_labels = emotion_labels
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
            else:
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)

            disp.plot(cmap='Blues', values_format='d')
            plt.title(f'{self.model_type.upper()} Model Confusion Matrix')
            plt.tight_layout()

            plt_path = f'results/{self.model_type}_confusion_matrix.png'
            plt.savefig(plt_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved to {plt_path}")
            plt.close()

        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")
            
    def _plot_per_class_metrics(self, report, emotion_labels=None):
        """Plot per-class metrics with proper error handling."""
        try:
            # Extract per-class metrics
            classes = []
            precision = []
            recall = []
            f1_score = []
            
            for i, label in enumerate(emotion_labels):
                if str(i) in report:
                    classes.append(label)
                    precision.append(report[str(i)]['precision'])
                    recall.append(report[str(i)]['recall'])
                    f1_score.append(report[str(i)]['f1-score'])
            
            # Create a DataFrame for easier plotting
            df = pd.DataFrame({
                'Class': classes,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1_score
            })
            
            # Melt the DataFrame for easier plotting
            df_melted = pd.melt(df, id_vars=['Class'], value_vars=['Precision', 'Recall', 'F1-Score'],
                               var_name='Metric', value_name='Value')
            
            # Plot
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Class', y='Value', hue='Metric', data=df_melted)
            plt.title(f'{self.model_type.upper()} Model - Per-Class Metrics')
            plt.xticks(rotation=45)
            plt.ylim(0, 1.0)
            plt.tight_layout()
            
            # Save plot
            plt_path = f'results/{self.model_type}_per_class_metrics.png'
            plt.savefig(plt_path, dpi=300, bbox_inches='tight')
            logger.info(f"Per-class metrics plot saved to {plt_path}")
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting per-class metrics: {e}")
            
    def _plot_prediction_distribution(self, y_true, y_pred, emotion_labels=None):
        """Plot prediction distribution with proper error handling."""
        try:
            # Count occurrences of each class in true and predicted labels
            true_counts = np.bincount(y_true, minlength=len(emotion_labels) if emotion_labels else None)
            pred_counts = np.bincount(y_pred, minlength=len(emotion_labels) if emotion_labels else None)
            
            # Create a DataFrame for plotting
            df = pd.DataFrame({
                'Emotion': emotion_labels if emotion_labels else [f'Class {i}' for i in range(len(true_counts))],
                'True Count': true_counts,
                'Predicted Count': pred_counts
            })
            
            # Melt the DataFrame for easier plotting
            df_melted = pd.melt(df, id_vars=['Emotion'], value_vars=['True Count', 'Predicted Count'],
                               var_name='Type', value_name='Count')
            
            # Plot
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Emotion', y='Count', hue='Type', data=df_melted)
            plt.title(f'{self.model_type.upper()} Model - Prediction Distribution')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            plt_path = f'results/{self.model_type}_prediction_distribution.png'
            plt.savefig(plt_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction distribution plot saved to {plt_path}")
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting prediction distribution: {e}")
            
    def _plot_training_history(self):
        """Plot training history with proper error handling."""
        try:
            if self.history is None:
                logger.warning("No training history available to plot")
                return
                
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot accuracy
            ax1.plot(self.history.history['accuracy'], label='Training')
            ax1.plot(self.history.history['val_accuracy'], label='Validation')
            ax1.set_title('Model Accuracy')
            ax1.set_ylabel('Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.legend()
            ax1.grid(True)
            
            # Plot loss
            ax2.plot(self.history.history['loss'], label='Training')
            ax2.plot(self.history.history['val_loss'], label='Validation')
            ax2.set_title('Model Loss')
            ax2.set_ylabel('Loss')
            ax2.set_xlabel('Epoch')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt_path = f'results/{self.model_type}_training_history.png'
            plt.savefig(plt_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {plt_path}")
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting training history: {e}")

if __name__ == "__main__":
    from emotion_model import EmotionModel
    
    # Create a simple model for testing
    emotion_model = EmotionModel(num_classes=7)
    mlp_model = emotion_model.build_mlp(input_shape=(13,))
    
    # Create trainer
    trainer = ModelTrainer(mlp_model, model_type='mlp')
    
    # Generate dummy data for testing with explicit dtypes
    X_train = np.random.random((100, 13)).astype(np.float32)
    y_train = np.random.randint(0, 7, size=(100,), dtype=np.int32)
    X_val = np.random.random((20, 13)).astype(np.float32)
    y_val = np.random.randint(0, 7, size=(20,), dtype=np.int32)
    X_test = np.random.random((30, 13)).astype(np.float32)
    y_test = np.random.randint(0, 7, size=(30,), dtype=np.int32)
    
    # Train model
    trainer.train(X_train, y_train, X_val, y_val, epochs=5)
    
    # Evaluate model
    metrics = trainer.evaluate(X_test, y_test)
    
    # Save model
    trainer.save_model('results/mlp_model.h5')