from .emotion_model import EmotionModel
"""Convenience exports for the models package.

This allows imports like:

    from src.models import EmotionModel, ModelManager, ModelTrainer, GeneticOptimizer

"""

from .emotion_model import EmotionModel
from .model_manager import ModelManager
from .trainer import ModelTrainer
from .optimizer import GeneticOptimizer

__all__ = [
    "EmotionModel",
    "ModelManager",
    "ModelTrainer",
    "GeneticOptimizer",
]

__all__ = ['EmotionModel']