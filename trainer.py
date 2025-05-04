import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
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
        Train the model on the provided data.
        
        Args:
            X_train (numpy.ndarray): Training features.
            y_train (numpy.ndarray): Training labels.
            X_val (numpy.ndarray): Validation features.
            y_val (numpy.ndarray): Validation labels.
            batch_size (int): Batch size for training.
            epochs (int): Maximum number of epochs to train for.
            callbacks (list): List of callbacks for training.
            
        Returns:
            tensorflow.keras.callbacks.History: Training history.
        """
        try:
            if callbacks is None:
                callbacks = [
                    EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True,
                        verbose=1
                    )
                ]
            
            logger.info(f"Starting {self.model_type.upper()} model training for {epochs} epochs with batch size {batch_size}")
            
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            logger.info(f"Model training completed after {len(self.history.history['loss'])} epochs")
            
            return self.history
        
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def evaluate(self, X_test, y_test, emotion_labels=None):
        """
        Evaluate the model on the test data.
        
        Args:
            X_test (numpy.ndarray): Test features.
            y_test (numpy.ndarray): Test labels.
            emotion_labels (list): List of emotion label names.
            
        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        try:
            if emotion_labels is None:
                emotion_labels = ['calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
            
            logger.info(f"Evaluating {self.model_type.upper()} model on test data")
            
            # Get model predictions
            y_pred_proba = self.model.predict(X_test)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
            precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            # Create confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Store metrics in a dictionary
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'precision_avg': precision_avg,
                'recall_avg': recall_avg,
                'f1_avg': f1_avg,
                'confusion_matrix': cm
            }
            
            # Log metrics
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Average Precision: {precision_avg:.4f}")
            logger.info(f"Average Recall: {recall_avg:.4f}")
            logger.info(f"Average F1-score: {f1_avg:.4f}")
            
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix - {self.model_type.upper()}')
            
            # Create output directory if it doesn't exist
            os.makedirs('results', exist_ok=True)
            plt.savefig(f'results/{self.model_type}_confusion_matrix.png')
            
            # Plot training history if available
            if self.history is not None:
                plt.figure(figsize=(12, 5))
                
                # Plot accuracy
                plt.subplot(1, 2, 1)
                plt.plot(self.history.history['accuracy'], label='Train')
                plt.plot(self.history.history['val_accuracy'], label='Validation')
                plt.title(f'Model Accuracy - {self.model_type.upper()}')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                
                # Plot loss
                plt.subplot(1, 2, 2)
                plt.plot(self.history.history['loss'], label='Train')
                plt.plot(self.history.history['val_loss'], label='Validation')
                plt.title(f'Model Loss - {self.model_type.upper()}')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(f'results/{self.model_type}_training_history.png')
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Args:
            filepath (str): Path to save the model.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
        
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