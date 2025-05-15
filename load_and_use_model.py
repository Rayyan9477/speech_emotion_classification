#!/usr/bin/env python3
"""
load_and_use_model.py - Load a pre-trained model for inference
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from datetime import datetime

print("TensorFlow version:", tf.__version__)

# Create directories
os.makedirs('results', exist_ok=True)

# Constants
NUM_CLASSES = 7
INPUT_SHAPE = (20, 20, 1)
EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust']

# Generate test data
print("Generating test data...")
X_test = np.random.random((30, *INPUT_SHAPE)).astype(np.float32)
y_test = np.random.randint(0, NUM_CLASSES, size=(30,), dtype=np.int32)

# Look for existing models 
model_path = None
for model_dir in ['models', 'logs']:
    if os.path.exists(model_dir):
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if file.endswith(('.keras', '.h5')) and 'best_model' in file:
                    model_path = os.path.join(root, file)
                    break
            if model_path:
                break
    if model_path:
        break

# If model found, load it
if model_path:
    print(f"Loading model from {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
        
        # Try to use the model for prediction
        print("Running inference on test data...")
        
        # Get predictions
        predictions = model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Evaluate accuracy
        accuracy = np.mean(predicted_classes == y_test)
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Save results
        results = {
            "model_path": model_path,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "accuracy": float(accuracy),
            "predictions": [int(pred) for pred in predicted_classes],
            "ground_truth": [int(true) for true in y_test]
        }
        
        # Save results
        with open('results/inference_results.json', 'w') as f:
            json.dump(results, f, indent=4)
            
        print("Inference results saved to results/inference_results.json")
        
    except Exception as e:
        print(f"Error loading or using model: {e}")
else:
    print("No pre-trained model found. Creating a new model...")
    
    # Create a simplified functional model 
    inputs = tf.keras.layers.Input(shape=INPUT_SHAPE)
    x = tf.keras.layers.Conv2D(8, kernel_size=(3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile with a different optimizer to avoid overflow
    model.compile(
        optimizer='rmsprop',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Save the model without training
    model_save_path = os.path.join('models', 'untrained_model.keras')
    model.save(model_save_path)
    
    print(f"Untrained model saved to {model_save_path}")
    print("Note: You need to train this model with another script before it's useful for inference.")
