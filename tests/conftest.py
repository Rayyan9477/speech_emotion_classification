"""
conftest.py - Shared test fixtures for the speech emotion classification system.
"""

import pytest
import numpy as np
import os
import tensorflow as tf
from pathlib import Path

@pytest.fixture
def sample_audio_data():
    """Generate sample audio data for testing."""
    return np.random.random(16000)  # 1 second of audio at 16kHz

@pytest.fixture
def sample_mfcc_features():
    """Generate sample MFCC features for testing."""
    return np.random.random((13,))  # 13 MFCC coefficients

@pytest.fixture
def sample_spectrogram():
    """Generate sample spectrogram features for testing."""
    return np.random.random((128, 100, 1))  # (n_mels, time_steps, channels)

@pytest.fixture
def sample_model():
    """Create a simple model for testing."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(13,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

@pytest.fixture
def temp_model_path(tmp_path):
    """Create a temporary path for saving/loading models."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return str(model_dir / "test_model.keras")

@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return {
        'train': {
            'audio': [
                {'path': f'/tmp/audio_{i}.wav', 'array': np.random.random(16000)}
                for i in range(10)
            ],
            'labels': np.random.randint(0, 7, size=10)
        }
    }

@pytest.fixture
def test_dirs(tmp_path):
    """Create temporary directories for testing."""
    dirs = {
        'models': tmp_path / "models",
        'results': tmp_path / "results",
        'logs': tmp_path / "logs",
        'data': tmp_path / "data"
    }
    for dir_path in dirs.values():
        dir_path.mkdir()
    return dirs
