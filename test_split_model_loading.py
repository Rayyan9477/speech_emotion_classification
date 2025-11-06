#!/usr/bin/env python3
"""Test script to verify split model loading functionality"""

import os
import sys
import tempfile
from pathlib import Path

def load_split_model_from_manifest(manifest_path):
    """Load a model that has been split into multiple parts.
    
    Args:
        manifest_path (str): Path to the manifest file
        
    Returns:
        str: Path to the combined model file, or None if failed
    """
    try:
        manifest_path = Path(manifest_path)
        models_dir = manifest_path.parent
        
        print(f"✓ Loading split model from manifest: {manifest_path}")
        
        # Read manifest to get chunk information
        with open(manifest_path, 'r') as f:
            lines = f.readlines()
        
        # Parse manifest
        original_filename = None
        chunk_files = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('Original file:'):
                original_filename = line.split(':', 1)[1].strip()
            elif line.startswith('- '):
                chunk_file = line[2:].strip()
                chunk_path = models_dir / chunk_file
                if chunk_path.exists():
                    chunk_files.append(str(chunk_path))
                    print(f"  ✓ Found chunk: {chunk_file}")
                else:
                    print(f"  ✗ Chunk file not found: {chunk_path}")
                    return None
        
        if not original_filename:
            print("✗ No original filename found in manifest")
            return None
            
        if not chunk_files:
            print("✗ No chunk files found")
            return None
        
        print(f"✓ Found {len(chunk_files)} chunks for {original_filename}")
        
        # Create temporary file to combine chunks
        temp_dir = tempfile.gettempdir()
        combined_path = os.path.join(temp_dir, original_filename)
        
        print(f"✓ Combining chunks to: {combined_path}")
        
        # Combine chunks
        with open(combined_path, 'wb') as output_file:
            for i, chunk_path in enumerate(chunk_files, 1):
                print(f"  → Combining chunk {i}/{len(chunk_files)}: {Path(chunk_path).name}")
                with open(chunk_path, 'rb') as chunk_file:
                    chunk_data = chunk_file.read()
                    output_file.write(chunk_data)
                    print(f"    Added {len(chunk_data):,} bytes")
        
        file_size_mb = os.path.getsize(combined_path) / (1024 * 1024)
        print(f"✓ Successfully combined model: {combined_path} ({file_size_mb:.1f} MB)")
        
        # Verify the file exists and has content
        if os.path.exists(combined_path) and os.path.getsize(combined_path) > 0:
            print(f"✓ Verification passed: File exists with size {file_size_mb:.1f} MB")
            return combined_path
        else:
            print("✗ Verification failed: File is empty or doesn't exist")
            return None
        
    except Exception as e:
        print(f"✗ Error loading split model: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_tensorflow_loading(model_path):
    """Test if TensorFlow can load the combined model"""
    try:
        import tensorflow as tf
        print(f"\n✓ TensorFlow version: {tf.__version__}")
        print(f"✓ Attempting to load model from: {model_path}")
        
        model = tf.keras.models.load_model(model_path)
        print(f"✓ Model loaded successfully!")
        print(f"  Model type: {type(model)}")
        print(f"  Input shape: {model.input_shape if hasattr(model, 'input_shape') else 'Multiple inputs'}")
        print(f"  Output shape: {model.output_shape if hasattr(model, 'output_shape') else 'Multiple outputs'}")
        
        if hasattr(model, 'inputs'):
            print(f"  Number of inputs: {len(model.inputs)}")
            for i, inp in enumerate(model.inputs):
                print(f"    Input {i}: {inp.shape}")
        
        if hasattr(model, 'outputs'):
            print(f"  Number of outputs: {len(model.outputs)}")
            for i, out in enumerate(model.outputs):
                print(f"    Output {i}: {out.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading model with TensorFlow: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("SPLIT MODEL LOADING TEST")
    print("=" * 60)
    
    # Test manifest file
    manifest_path = "models/cnn_emotion_model_20251022_065208_manifest.txt"
    
    if not os.path.exists(manifest_path):
        print(f"✗ Manifest file not found: {manifest_path}")
        return False
    
    print(f"✓ Found manifest file: {manifest_path}\n")
    
    # Test combining chunks
    combined_path = load_split_model_from_manifest(manifest_path)
    
    if not combined_path:
        print("\n✗ FAILED: Could not combine model chunks")
        return False
    
    print(f"\n{'=' * 60}")
    print("TESTING TENSORFLOW MODEL LOADING")
    print("=" * 60)
    
    # Test TensorFlow loading
    success = test_tensorflow_loading(combined_path)
    
    # Cleanup
    if os.path.exists(combined_path):
        print(f"\n✓ Cleaning up temporary file: {combined_path}")
        os.remove(combined_path)
    
    print(f"\n{'=' * 60}")
    if success:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ TESTS FAILED!")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
