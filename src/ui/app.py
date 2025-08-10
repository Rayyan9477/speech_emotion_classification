"""
Thin shim to expose `EmotionAnalyzer` from the root-level `app.py`.
This keeps imports like `from src.ui.app import EmotionAnalyzer` working.
"""
from pathlib import Path
import sys

# Ensure project root is on sys.path so `import app` works
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app import EmotionAnalyzer  # noqa: F401
