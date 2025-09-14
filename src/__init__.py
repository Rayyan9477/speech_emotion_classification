"""
Speech Emotion Classification Package

This package provides tools for classifying emotions from speech audio using
neural networks and machine learning techniques.
"""

import logging
from pathlib import Path

# Configure package-level logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Apply monkey patch on import to fix TensorFlow issues
try:
    from .utils.monkey_patch import monkeypatch
    monkeypatch()
    logging.getLogger(__name__).info("TensorFlow monkey patch applied successfully")
except ImportError:
    logging.getLogger(__name__).warning("Could not apply TensorFlow monkey patch")
except Exception as e:
    logging.getLogger(__name__).warning(f"Error applying TensorFlow monkey patch: {e}")

__version__ = "0.1.0"
__author__ = "Speech Emotion Classification Team"