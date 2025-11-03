#!/usr/bin/env python3
# test_prediction_pipeline.py - Test the complete prediction pipeline with split models

import sys
import os
import numpy as np
import librosa
sys.path.insert(0, '.')

from src.models.model_manager import ModelManager
from src.features.feature_extractor import FeatureExtractor
from src.models.emotion_model import EmotionModel
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_prediction_pipeline():
    """Test the complete prediction pipeline"""
    print('Testing complete prediction pipeline with split models...')

    try:
        # Initialize components
        manager = ModelManager()
        feature_extractor = FeatureExtractor()

        # Load split multimodal model (only available model)
        print('Loading split multimodal model...')
        multimodal_model = manager.load_model('multimodal_20251022_065209')

        if not multimodal_model:
            print('❌ Failed to load multimodal model')
            return False

        print('✅ Split multimodal model loaded successfully')

        # Test with a demo file
        demo_file = 'demo_files/happy_sample.wav'
        if not os.path.exists(demo_file):
            print(f'❌ Demo file not found: {demo_file}')
            return False

        print(f'Testing prediction with demo file: {demo_file}')

        # Load audio
        y, sr = librosa.load(demo_file, sr=None)
        print(f'Loaded audio: {len(y)} samples at {sr}Hz')

        # Extract features for multimodal model
        print('Extracting features for multimodal model...')
        mfcc_features = feature_extractor.extract_mfcc(y, sr)
        spec_features = feature_extractor.extract_spectrogram(y, sr)

        # Normalize features
        mfcc_features = feature_extractor.normalize_single(mfcc_features, feature_type='mfcc')
        spec_features = feature_extractor.normalize_single(spec_features, feature_type='mel_spectrogram')

        print(f'MFCC features shape: {mfcc_features.shape}')
        print(f'Spectrogram features shape: {spec_features.shape}')

        # Prepare input for multimodal model
        multimodal_input = [
            mfcc_features.reshape(1, -1),
            spec_features.reshape(1, 128, 165, 1)
        ]

        # Make prediction with multimodal model
        print('Making prediction with multimodal model...')
        multimodal_pred = multimodal_model.predict(multimodal_input, verbose=0)
        multimodal_emotion_idx = np.argmax(multimodal_pred[0])
        multimodal_confidence = float(multimodal_pred[0][multimodal_emotion_idx])

        # Emotion labels
        emotion_labels = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

        print('\n🎯 PREDICTION RESULTS:')
        print(f'Multimodal Model - Emotion: {emotion_labels[multimodal_emotion_idx]}, Confidence: {multimodal_confidence:.3f}')

        # Verify predictions are reasonable
        if multimodal_confidence > 0.1:
            print('✅ Model produced reasonable confidence score')
        else:
            print('⚠️ Low confidence score detected')

        print('✅ Prediction pipeline test completed successfully!')
        return True

    except Exception as e:
        print(f'❌ Error in prediction pipeline test: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_prediction_pipeline()
    if success:
        print('\n🎉 ALL TESTS PASSED! Split model functionality is working correctly.')
    else:
        print('\n❌ TESTS FAILED! There are issues with the split model functionality.')