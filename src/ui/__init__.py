"""UI package exports.

Avoid importing root-level `EmotionAnalyzer` here to prevent circular imports
when `app.py` itself imports modules from `src.ui`.
"""
from .speech_emotion_analyzer import SpeechEmotionAnalyzer  # noqa: F401
from .dashboard import EmotionDashboard  # noqa: F401

__all__ = ["SpeechEmotionAnalyzer", "EmotionDashboard"]

