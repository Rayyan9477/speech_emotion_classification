import pytest
from pathlib import Path
try:
    from src.ui.app import EmotionAnalyzer
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'ui')))
    from app import EmotionAnalyzer

@pytest.fixture
def temp_upload_dir(tmp_path):
    return tmp_path / "uploads"

def test_upload_dir_creation(temp_upload_dir):
    # Test normal directory creation
    analyzer = EmotionAnalyzer()
    assert Path(analyzer.upload_folder).exists(), "Upload directory should be created"

def test_permission_fallback(monkeypatch, temp_upload_dir):
    # Simulate permission error
    def mock_mkdir(*args, **kwargs):
        raise PermissionError("Simulated permission error")
    
    monkeypatch.setattr(Path, "mkdir", mock_mkdir)
    
    analyzer = EmotionAnalyzer()
    assert "speech_emotion_uploads" in str(analyzer.upload_folder), \
        "Should fallback to temp directory on permission error"

def test_write_permissions(temp_upload_dir):
    analyzer = EmotionAnalyzer()
    test_file = Path(analyzer.upload_folder) / "test.txt"
    
    try:
        test_file.write_text("test")
        assert test_file.exists(), "Should be able to write in upload directory"
    finally:
        test_file.unlink(missing_ok=True)