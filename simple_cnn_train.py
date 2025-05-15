#!/usr/bin/env python3
"""
simple_cnn_train.py - Simplified CNN model training script for speech emotion classification
"""

import os
import sys
import time
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Monkey patch to fix TensorFlow OverflowError
def monkey_patch_tf():
    """Apply monkey patch to fix TensorFlow overflow error"""
    try:
        import tensorflow as tf
        import numpy as np
        
        # Find where the overflow occurs in Keras backend
        if 'keras.src.backend.tensorflow.numpy' in sys.modules:
            numpy_module = sys.modules['keras.src.backend.tensorflow.numpy']
            
            # Get the original argmax function
            if hasattr(numpy_module, 'argmax'):
                original_argmax = numpy_module.argmax
                
                # Define a new argmax that avoids the issue
                def safe_argmax(x, axis=None, keepdims=False):
                    """Safe implementation of argmax that doesn't use signbit."""
                    x = numpy_module.convert_to_tensor(x)
                    dtype = numpy_module.standardize_dtype(x.dtype)
                    if "float" not in dtype or x.ndim == 0:
                        _x = x
                        if axis is None:
                            x = tf.reshape(x, [-1])
                        y = tf.argmax(x, axis=axis, output_type="int32")
                        if keepdims:
                            y = numpy_module._keepdims(_x, y, axis)
                        return y
                    
                    # Fix for float types without using signbit
                    dtype = numpy_module.dtypes.result_type(dtype, "float32")
                    x = numpy_module.cast(x, dtype)
                    
                    # Handle -0.0 differently to avoid using problematic operations
                    eps = np.finfo(np.float32).tiny
                    zero_mask = tf.equal(x, 0.0)
                    
                    # Detect negative zeros safely without using signbit
                    # Add a tiny value to zeros to distinguish +0 from -0
                    x = tf.where(zero_mask, x + eps, x)
                    
                    _x = x
                    if axis is None:
                        x = tf.reshape(x, [-1])
                    y = tf.argmax(x, axis=axis, output_type="int32")
                    if keepdims:
                        y = numpy_module._keepdims(_x, y, axis)
                    return y
                
                # Replace the original argmax function
                numpy_module.argmax = safe_argmax
                return "Keras backend numpy.argmax function patched successfully"
        
        return "No patching required"
    except Exception as e:
        return f"Patching failed: {e}"

# Apply the monkey patch from the utils module
try:
    # Try to import the proper monkey patch from utils
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from src.utils.monkey_patch import monkeypatch
    patch_result = monkeypatch()
    print(f"Monkey patch applied: {patch_result}")
except Exception as e:
    print(f"Warning: Could not apply monkey patch from utils: {e}")
    # Fall back to local monkey patch
    try:
        patch_result = monkey_patch_tf()
        print(f"Local monkey patch applied: {patch_result}")
    except Exception as e2:
        print(f"Warning: Could not apply local monkey patch: {e2}")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simple_cnn_train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
MODEL_TYPE = 'cnn'
NUM_CLASSES = 7
EPOCHS = 50
BATCH_SIZE = 32
PATIENCE = 10  # For early stopping
EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust']

# Create directories
for directory in ['models', 'results', 'logs']:
    os.makedirs(directory, exist_ok=True)


def generate_dummy_data():
    """Generate dummy data for demonstration with smaller dimensions"""
    logger.info("Generating random data for demonstration.")
    
    # For CNN models with manageable dimensions: (samples, height, width, channels)
    X_train = np.random.random((200, 32, 32, 1)).astype(np.float32)
    y_train = np.random.randint(0, NUM_CLASSES, size=(200,), dtype=np.int32)
    
    X_val = np.random.random((50, 32, 32, 1)).astype(np.float32)
    y_val = np.random.randint(0, NUM_CLASSES, size=(50,), dtype=np.int32)
    
    X_test = np.random.random((50, 32, 32, 1)).astype(np.float32)
    y_test = np.random.randint(0, NUM_CLASSES, size=(50,), dtype=np.int32)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_cnn_model(input_shape, num_classes=7):
    """Build a smaller CNN model for emotion classification"""
    model = models.Sequential([
        # First convolutional block - smaller filters
        layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.3),
        
        # Second convolutional block - smaller filters
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.3),
        
        # Flatten layer
        layers.Flatten(),
        
        # Fully connected layers - smaller units
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Use a simpler optimizer with a lower learning rate
    optimizer = optimizers.Adam(learning_rate=0.0005)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_callbacks(patience=5):
    """Get callbacks for training"""
    # Create a unique log directory for each run
    run_id = datetime.now().strftime('run_%Y%m%d_%H%M%S')
    log_dir = os.path.join('logs', run_id)
    os.makedirs(log_dir, exist_ok=True)
    return [
        # Model checkpoint to save the best model
        callbacks.ModelCheckpoint(
            filepath=os.path.join(log_dir, 'best_model.keras'),
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        # Reduce learning rate when training plateaus
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1
        ),
        # TensorBoard logging
        callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            update_freq='epoch',
            profile_batch=0
        )
    ]


def plot_training_history(history, model_type='cnn'):
    """Plot and save training history"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training')
        ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training')
        ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt_path = f'results/{model_type}_training_history.png'
        plt.savefig(plt_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {plt_path}")
        plt.close()
        
        # Also save history as JSON
        import json
        history_dict = {
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }
        with open(f'results/{model_type}_training_history.json', 'w') as f:
            json.dump(history_dict, f, indent=4)
        
    except Exception as e:
        logger.error(f"Error plotting training history: {e}")


def save_model_info(model_path, metrics, model_type='cnn'):
    """Save model info to JSON file"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create model info dictionary
        model_info = {
            "model_path": model_path,
            "model_type": model_type,
            "created": timestamp,
            "metrics": metrics,
            "description": f"{model_type.upper()} model for speech emotion classification"
        }
        
        # Save to JSON file
        info_path = os.path.join('models', 'model_info.json')
        with open(info_path, 'w') as f:
            import json
            json.dump(model_info, f, indent=4)
            
        logger.info(f"Model info saved to {info_path}")
        
    except Exception as e:
        logger.error(f"Error saving model info: {e}")


def train_and_evaluate():
    """Train and evaluate the CNN model with improved error handling"""
    try:
        # Generate dummy data
        X_train, y_train, X_val, y_val, X_test, y_test = generate_dummy_data()
        
        # Log data shapes
        logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
        logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        
        # Build model
        input_shape = X_train.shape[1:]  # (height, width, channels)
        model = build_cnn_model(input_shape, num_classes=NUM_CLASSES)
        
        # Print model summary with error handling
        try:
            stringlist = []
            model.summary(print_fn=lambda x: stringlist.append(x))
            model_summary = "\n".join(stringlist)
            logger.info(f"Model Summary:\n{model_summary}")
        except Exception as e:
            logger.warning(f"Could not print model summary: {e}")
        
        # Get callbacks with reduced complexity
        training_callbacks = [
            # Reduce learning rate when training plateaus
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=PATIENCE // 2,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Train model
        logger.info(f"Starting {MODEL_TYPE.upper()} model training for {EPOCHS} epochs")
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=get_callbacks(patience=PATIENCE),
            verbose=1
        )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        # Save model
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join('models', f"{MODEL_TYPE}_emotion_model.keras")
            model.save(model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Plot training history
            plot_training_history(history, model_type=MODEL_TYPE)
            
            # Evaluate model
            logger.info("Evaluating model on test data")
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            logger.info(f"Test loss: {loss:.4f}")
            logger.info(f"Test accuracy: {accuracy:.4f}")
            
            metrics = {
                'accuracy': float(accuracy),
                'loss': float(loss)
            }
            
            # Calculate additional metrics with error handling
            try:
                from sklearn.metrics import classification_report, confusion_matrix
                y_pred = model.predict(X_test)
                y_pred_classes = np.argmax(y_pred, axis=1)
                
                report = classification_report(y_test, y_pred_classes, output_dict=True)
                
                metrics.update({
                    'precision_avg': float(report['weighted avg']['precision']),
                    'recall_avg': float(report['weighted avg']['recall']),
                    'f1_avg': float(report['weighted avg']['f1-score'])
                })
            except Exception as e:
                logger.error(f"Error calculating additional metrics: {e}")
                metrics.update({
                    'precision_avg': 0.0,
                    'recall_avg': 0.0,
                    'f1_avg': 0.0
                })
            
            # Save model info
            save_model_info(model_path, metrics, MODEL_TYPE)
            
            return model_path, metrics
            
        except Exception as e:
            logger.error(f"Error saving model or results: {e}")
            return None, None
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None


if __name__ == "__main__":
    # Check TensorFlow version
    logger.info(f"TensorFlow version: {tf.__version__}")
    
    # Check for GPU support
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"TensorFlow detected {len(gpus)} GPU(s): {gpus}")
        # Allow memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        logger.warning("No GPU detected. Training will be slower on CPU.")
    
    # Train and evaluate model
    model_path, metrics = train_and_evaluate()
    
    if model_path and metrics:
        logger.info("===== Training Summary =====")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_avg']:.4f}")
        logger.info("============================")
    else:
        logger.error("Training failed.")
