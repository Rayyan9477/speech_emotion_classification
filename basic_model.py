#!/usr/bin/env python3
"""
basic_model.py - Very basic model training script for speech emotion classification
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("basic_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the src directory to the Python path if needed
project_root = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(project_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import monkey patch to fix TensorFlow overflow issues
try:
    from src.utils.monkey_patch import monkeypatch
    if monkeypatch():
        logger.info("Successfully applied TensorFlow monkey patch")
    else:
        logger.warning("Failed to apply TensorFlow monkey patch")
except ImportError:
    logger.warning("Could not import monkey_patch module, some TensorFlow operations may fail")

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Constants
NUM_CLASSES = 7
INPUT_SHAPE = (20, 20, 1)  # Very small input shape
BATCH_SIZE = 16
EPOCHS = 30
EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust']

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Set up environment and create directories
try:
    from src.utils.training_utils import setup_environment, create_model_directories, generate_dummy_data
    
    # Set up TensorFlow and environment
    setup_environment()
    
    # Create necessary directories
    create_model_directories()
    
    # Generate dummy data using the utility function
    X_train, y_train, X_val, y_val, X_test, y_test = generate_dummy_data(
        num_classes=NUM_CLASSES,
        input_shape=INPUT_SHAPE,
        train_samples=100,
        val_samples=30,
        test_samples=30
    )
    
except ImportError:
    logger.warning("Could not import training_utils, using basic implementation")
    # Generate very small dummy data
    logger.info("Generating dummy data...")
    X_train = np.random.random((100, *INPUT_SHAPE)).astype(np.float32)
    y_train = np.random.randint(0, NUM_CLASSES, size=(100,), dtype=np.int32)

    X_val = np.random.random((30, *INPUT_SHAPE)).astype(np.float32)
    y_val = np.random.randint(0, NUM_CLASSES, size=(30,), dtype=np.int32)

    X_test = np.random.random((30, *INPUT_SHAPE)).astype(np.float32)
    y_test = np.random.randint(0, NUM_CLASSES, size=(30,), dtype=np.int32)

    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# Try to import our model classes
try:
    from src.models.emotion_model import EmotionModel
    from src.models.trainer import ModelTrainer
    use_custom_classes = True
    logger.info("Using custom model classes from src.models")
except ImportError as e:
    logger.warning(f"Could not import custom model classes: {e}. Using basic implementation.")
    use_custom_classes = False

# Create a very small model
logger.info("Creating model...")

if use_custom_classes:
    # Use our custom EmotionModel class
    emotion_model = EmotionModel(num_classes=NUM_CLASSES)
    model = emotion_model.build_cnn(input_shape=INPUT_SHAPE)
    
    # Get the trainer ready
    trainer = ModelTrainer(model=model, model_type='cnn')
    
    # Get callbacks from our custom implementation
    callbacks = emotion_model.get_callbacks(patience=10)
    
    logger.info("Using custom EmotionModel and ModelTrainer classes")
else:
    # Fallback to basic implementation
    model = keras.Sequential([
        keras.layers.Input(shape=INPUT_SHAPE),
        keras.layers.Conv2D(8, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# Print model summary
model.summary(print_fn=logger.info if use_custom_classes else print)

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Create callbacks for training if not using custom classes
if not use_custom_classes:
    # Create directories if they don't exist
    os.makedirs('logs', exist_ok=True)
    
    # Model checkpoint to save the best model
    checkpoint_path = os.path.join('models', "basic_best_model.keras")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    # Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,  # Wait for 10 epochs before stopping if no improvement
        restore_best_weights=True,
        verbose=1
    )

    # Add TensorBoard callback for visualization
    log_dir = os.path.join('logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True
    )
    
    # Combine callbacks
    callbacks = [early_stopping, checkpoint, tensorboard_callback]

# Train model
logger.info(f"Training model for {EPOCHS} epochs...")
start_time = time.time()

try:
    if use_custom_classes:
        # Use our custom trainer
        history = trainer.train(
            X_train=X_train, 
            y_train=y_train,
            X_val=X_val, 
            y_val=y_val,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=callbacks
        )
    else:
        # Use basic training approach
        history = model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=2  # Less output
        )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Save training history as plot
    try:
        from src.utils.training_utils import save_training_history
        history_plot_path = save_training_history(history, filename='basic_model_training_history.png')
        logger.info(f"Training history plot saved to {history_plot_path}")
    except ImportError:
        # Fallback to basic implementation
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/training_history.png')
        logger.info("Training history plot saved to results/training_history.png")
    
    # Evaluate model
    logger.info("Evaluating model...")
    
    if use_custom_classes:
        # Use our custom evaluation
        metrics = trainer.evaluate(
            X_test=X_test, 
            y_test=y_test,
            emotion_labels=EMOTION_LABELS
        )
        loss = metrics['loss']
        accuracy = metrics['accuracy']
    else:
        # Basic evaluation
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    logger.info(f"Test loss: {loss:.4f}")
    logger.info(f"Test accuracy: {accuracy:.4f}")
    
    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join('models', f'basic_emotion_model_{timestamp}.keras')
    
    if use_custom_classes:
        # Use our custom save method
        saved_path = trainer.save_model(model_path)
    else:
        # Basic save
        model.save(model_path)
        saved_path = model_path
        
    logger.info(f"Model saved to {saved_path}")
    
    # Save model info
    model_info = {
        "model_path": saved_path,
        "model_type": "cnn",
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_shape": INPUT_SHAPE,
        "metrics": {
            "accuracy": float(accuracy),
            "loss": float(loss)
        },
        "training_time": training_time,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "emotion_labels": EMOTION_LABELS
    }
    
    # Try to use model_manager if available
    try:
        if use_custom_classes:
            from src.models.model_manager import ModelManager
            model_manager = ModelManager()
            model_manager.register_model(
                model_path=saved_path,
                model_type="cnn",
                metrics=model_info["metrics"],
                description="Basic CNN model for speech emotion classification"
            )
            logger.info("Model registered with ModelManager")
    except ImportError:
        # Fallback to basic JSON storage
        with open('models/model_info.json', 'w') as f:
            json.dump(model_info, f, indent=4)
        logger.info("Model info saved to models/model_info.json")
    
    logger.info("Model training completed successfully!")
    
except Exception as e:
    logger.error(f"Error during training: {e}")
    import traceback
    logger.error(traceback.format_exc())
