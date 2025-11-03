#!/usr/bin/env python3
# test_split_models.py - Test split model loading functionality

import sys
import os
sys.path.insert(0, '.')

from src.models.model_manager import ModelManager
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_split_model_loading():
    """Test loading split models"""
    manager = ModelManager()

    print('Testing split model loading...')

    try:
        # Test multimodal split model (only available split model)
        print('Loading multimodal split model...')
        multimodal_model = manager.load_model('multimodal_20251022_065209')
        if multimodal_model:
            print('✅ Multimodal split model loaded successfully')
            print(f'Model type: {type(multimodal_model)}')
            print(f'Model inputs: {len(multimodal_model.inputs) if hasattr(multimodal_model, "inputs") else "N/A"}')
        else:
            print('❌ Failed to load multimodal split model')

    except Exception as e:
        print(f'❌ Error testing split model loading: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_split_model_loading()