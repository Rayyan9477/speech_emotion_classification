"""
Main Streamlit entry point.

We standardize on `src/ui/app.py:EmotionAnalyzer` as the primary UI.
This thin entrypoint ensures `streamlit run -m src.ui.streamlit_app` works.
"""

from src.ui.app import EmotionAnalyzer


def main():
    app = EmotionAnalyzer()
    app.run()


if __name__ == "__main__":
    main()
