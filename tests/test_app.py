#!/usr/bin/env python3
# test_app.py - Test if the speech emotion classification app works correctly

import os
import sys
import logging

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_emotion_analyzer_import():
    """Basic smoke test: can import and instantiate EmotionAnalyzer without training."""
    from app import EmotionAnalyzer  # noqa: WPS433
    analyzer = EmotionAnalyzer()
    assert analyzer.emotion_labels, "Emotion labels should be initialized"

if __name__ == "__main__":  # Manual quick check
    test_emotion_analyzer_import()
    print("Smoke test passed")
