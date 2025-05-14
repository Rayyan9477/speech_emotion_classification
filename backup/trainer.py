import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Class for training and evaluating speech emotion classification models.
    """
    def __init__(self, model, model_type='cnn'):
        """
        Initialize the ModelTrainer with a model.
        
        Args:
            model (tensorflow.keras.models.Model): The model to train and evaluate.
            model_type (str): Type of the model ('mlp' or 'cnn').
        """
        self.model = model
        self.model_type = model_type
        self.history = None
        
    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=50, callbacks=None):
        """
        Train the model.
        
        Args:
            X_train (numpy.ndarray): Training features.
            y_train (numpy.ndarray): Training labels.
            X_val (numpy.ndarray): Validation features.
            y_val (numpy.ndarray): Validation labels.
            batch_size (int): Batch size for training.
            epochs (int): Maximum number of epochs.
            callbacks (list): List of callbacks for training.
            
        Returns:
            dict: Training history.
        """
        try:
            logger.info(f"Starting {self.model_type.upper()} model training for {epochs} epochs with batch size {batch_size}")
            
            # Train the model
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            # Plot training history
            self._plot_training_history()
            
            return self.history
        
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

    def _plot_training_history(self):
        """
        Plot the training history (accuracy and loss).
        """
        try:
            if self.history is None:
                logger.warning("No training history available")
                return
            
            # Create figure with 2 subplots
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
            plt.savefig(f'results/{self.model_type}_training_history.png', dpi=300)
            logger.info(f"Training history plot saved to results/{self.model_type}_training_history.png")
            
        except Exception as e:
            logger.error(f"Error plotting training history: {e}")
            
    def evaluate(self, X_test, y_test, emotion_labels=None):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (numpy.ndarray): Test features.
            y_test (numpy.ndarray): Test labels.
            emotion_labels (list): List of emotion labels.
            
        Returns:
            dict: Evaluation metrics.
        """
        try:
            # Evaluate model
            loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
            logger.info(f"Test loss: {loss:.4f}")
            logger.info(f"Test accuracy: {accuracy:.4f}")
            
            # Get predictions
            y_pred = self.model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            # Calculate metrics
            report = classification_report(y_test, y_pred_classes, output_dict=True)
            
            # Plot confusion matrix
            self._plot_confusion_matrix(y_test, y_pred_classes, emotion_labels)
            
            # Create a detailed results dictionary
            metrics = {
                'accuracy': accuracy,
                'loss': loss,
                'precision_avg': report['weighted avg']['precision'],
                'recall_avg': report['weighted avg']['recall'],
                'f1_avg': report['weighted avg']['f1-score'],
                'report': report,
                'confusion_matrix': confusion_matrix(y_test, y_pred_classes)
            }
            
            # Save detailed metrics to a CSV file
            self._save_metrics_to_csv(metrics, emotion_labels)
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
            
    def _save_metrics_to_csv(self, metrics, emotion_labels):
        """
        Save detailed metrics to a CSV file.
        
        Args:
            metrics (dict): Evaluation metrics.
            emotion_labels (list): List of emotion labels.
        """
        try:
            if 'report' not in metrics:
                return
                
            # Create reports directory if it doesn't exist
            os.makedirs('results/reports', exist_ok=True)
            
            # Save per-class metrics
            class_metrics = []
            for i, label in enumerate(emotion_labels):
                if str(i) in metrics['report']:
                    class_metrics.append({
                        'emotion': label,
                        'precision': metrics['report'][str(i)]['precision'],
                        'recall': metrics['report'][str(i)]['recall'],
                        'f1_score': metrics['report'][str(i)]['f1-score'],
                        'support': metrics['report'][str(i)]['support']
                    })
            
            # Create DataFrame and save to CSV
            import pandas as pd
            metrics_df = pd.DataFrame(class_metrics)
            metrics_df.to_csv(f'results/reports/{self.model_type}_class_metrics.csv', index=False)
            logger.info(f"Class metrics saved to results/reports/{self.model_type}_class_metrics.csv")
            
        except Exception as e:
            logger.error(f"Error saving metrics to CSV: {e}")
    
    def _plot_confusion_matrix(self, y_true, y_pred, emotion_labels=None):
        """
        Plot confusion matrix.
        
        Args:
            y_true (numpy.ndarray): True labels.
            y_pred (numpy.ndarray): Predicted labels.
            emotion_labels (list): List of emotion labels.
        """
        try:
            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            
            # Use emotion labels if provided, otherwise use numeric labels
            if emotion_labels is not None:
                display_labels = emotion_labels
            else:
                display_labels = [str(i) for i in range(cm.shape[0])]
            
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=display_labels
            )
            
            disp.plot(cmap=plt.cm.Blues, values_format='.2f')
            plt.title(f'{self.model_type.upper()} Model Confusion Matrix')
            plt.tight_layout()
            plt.savefig(f'results/{self.model_type}_confusion_matrix.png', dpi=300)
            logger.info(f"Confusion matrix plot saved to results/{self.model_type}_confusion_matrix.png")
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")
    
    def save_model(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath (str): Path to save the model.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # If .h5 file is requested, provide a warning but save in that format
            if filepath.endswith('.h5'):
                logger.warning("Saving model in HDF5 format, which is considered legacy. Consider using .keras format instead.")
                self.model.save(filepath)
            else:
                # Use .keras extension for native Keras format
                if not filepath.endswith('.keras'):
                    filepath = f"{filepath}.keras"
                self.model.save(filepath)
            
            logger.info(f"Model saved to {filepath}")
            
            # Save model architecture as JSON for reference
            json_filepath = filepath.rsplit('.', 1)[0] + "_architecture.json"
            with open(json_filepath, 'w') as f:
                f.write(self.model.to_json())
            logger.info(f"Model architecture saved to {json_filepath}")
        
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath):
        """
        Load a trained model from a file.
        
        Args:
            filepath (str): Path to the saved model.
            
        Returns:
            tensorflow.keras.models.Model: The loaded model.
        """
        try:
            self.model = tf.keras.models.load_model(filepath)
            logger.info(f"Model loaded from {filepath}")
            return self.model
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    from model import EmotionModel
    
    # Create a simple model for testing
    emotion_model = EmotionModel(num_classes=7)
    mlp_model = emotion_model.build_mlp(input_shape=(13,))
    
    # Create trainer
    trainer = ModelTrainer(mlp_model, model_type='mlp')
    
    # Generate dummy data for testing
    X_train = np.random.random((100, 13))
    y_train = np.random.randint(0, 7, size=(100,))
    X_val = np.random.random((20, 13))
    y_val = np.random.randint(0, 7, size=(20,))
    X_test = np.random.random((30, 13))
    y_test = np.random.randint(0, 7, size=(30,))
    
    # Train model
    trainer.train(X_train, y_train, X_val, y_val, epochs=5)
    
    # Evaluate model
    metrics = trainer.evaluate(X_test, y_test)
    
    # Save model
    trainer.save_model('results/mlp_model.h5')