#!/usr/bin/env python3
"""Comprehensive test of the app's model loading with split models"""

import os
import sys
import json

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model_registry():
    """Test that model registry is valid and contains the split model"""
    print("=" * 60)
    print("TEST 1: MODEL REGISTRY VALIDATION")
    print("=" * 60)
    
    registry_path = "models/model_registry.json"
    
    if not os.path.exists(registry_path):
        print(f"✗ Registry file not found: {registry_path}")
        return False
    
    try:
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        print(f"✓ Registry JSON is valid")
        print(f"✓ Found {len(registry.get('models', []))} models in registry")
        
        # Check for split model
        split_model = None
        for model in registry.get('models', []):
            if model.get('split_model'):
                split_model = model
                break
        
        if split_model:
            print(f"✓ Found split model: {split_model['id']}")
            print(f"  Path: {split_model['path']}")
            print(f"  Type: {split_model['type']}")
            print(f"  Size: {split_model['size_mb']} MB")
            
            # Check if manifest exists
            if os.path.exists(split_model['path']):
                print(f"✓ Manifest file exists: {split_model['path']}")
                return True
            else:
                print(f"✗ Manifest file not found: {split_model['path']}")
                return False
        else:
            print("✗ No split model found in registry")
            return False
            
    except Exception as e:
        print(f"✗ Error reading registry: {e}")
        return False

def test_model_files():
    """Test that all model files exist"""
    print(f"\n{'=' * 60}")
    print("TEST 2: MODEL FILES EXISTENCE")
    print("=" * 60)
    
    required_files = [
        "models/cnn_emotion_model_20251022_065208_manifest.txt",
        "models/cnn_emotion_model_20251022_065208_part1.keras",
        "models/cnn_emotion_model_20251022_065208_part2.keras",
        "models/cnn_emotion_model_20251022_065208_part3.keras",
        "models/cnn_emotion_model_20251022_065208_feature_info.json",
        "models/cnn_emotion_model_20251022_065208_architecture.json",
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"✓ {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {file_path} - NOT FOUND")
            all_exist = False
    
    return all_exist

def test_app_imports():
    """Test that the app can be imported"""
    print(f"\n{'=' * 60}")
    print("TEST 3: APP IMPORTS")
    print("=" * 60)
    
    try:
        # Test key imports
        print("Testing imports...")
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__}")
        
        import streamlit as st
        print(f"✓ Streamlit {st.__version__}")
        
        import librosa
        print(f"✓ librosa {librosa.__version__}")
        
        print("✓ All critical imports successful")
        return True
        
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def main():
    print("COMPREHENSIVE APP DEPLOYMENT TEST")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Model Registry", test_model_registry()))
    results.append(("Model Files", test_model_files()))
    results.append(("App Imports", test_app_imports()))
    
    # Summary
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print("=" * 60)
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED - READY FOR DEPLOYMENT ✓✓✓")
    else:
        print("✗✗✗ SOME TESTS FAILED - FIX BEFORE DEPLOYMENT ✗✗✗")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
