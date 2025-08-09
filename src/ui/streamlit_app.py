"""
Main streamlit entry point for the Speech Emotion Recognition System.
"""

from src.ui.speech_emotion_analyzer import SpeechEmotionAnalyzer

def main():
    app = SpeechEmotionAnalyzer()
    app.run()

if __name__ == "__main__":
    main()
