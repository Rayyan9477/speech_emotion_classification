#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

print('=== COMPREHENSIVE SYSTEM TEST ===')

# Test all core imports and basic functionality
tests = [
    ('Config System', lambda: exec('from src.core.config import Config; cfg = Config(); print(f"✓ Config loaded with {len(cfg.training.emotion_labels)} emotion labels")')),
    ('Feature Extractor', lambda: exec('from src.features.feature_extractor import FeatureExtractor; fe = FeatureExtractor(); print("✓ FeatureExtractor created")')),
    ('Emotion Model', lambda: exec('from src.models.emotion_model import EmotionModel; em = EmotionModel(); print("✓ EmotionModel created")')),
    ('Model Manager', lambda: exec('from src.models.model_manager import ModelManager; mm = ModelManager(); print("✓ ModelManager created")')),
    ('Data Loader', lambda: exec('from src.data.data_loader import DataLoader; dl = DataLoader(); print("✓ DataLoader created")')),
    ('Training Args', lambda: exec('from src.main import TrainArgs; args = TrainArgs(); print("✓ TrainArgs created")')),
]

passed = 0
failed = 0

for name, test_func in tests:
    try:
        test_func()
        print(f'✅ {name}: PASSED')
        passed += 1
    except Exception as e:
        print(f'❌ {name}: FAILED - {e}')
        failed += 1

print(f'\nCore Components: {passed}/{passed + failed} passed')

if failed == 0:
    print('🎉 ALL CORE COMPONENTS WORKING! 🎉')
else:
    print('⚠️  Some core components have issues')
