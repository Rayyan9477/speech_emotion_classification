"""
Speech Emotion Classification Package

A deep learning system for classifying emotions in speech using various neural network architectures.
"""

import logging

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("speech_emotion.log"),
        logging.StreamHandler()
    ]
)

# Version of the speech_emotion_classification package
__version__ = "1.0.0"