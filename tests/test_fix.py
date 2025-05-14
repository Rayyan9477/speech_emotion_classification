#!/usr/bin/env python3
# test_fix.py - A simple script to test that our signbit fix works correctly

import os
import numpy as np
import tensorflow as tf
import logging
try:
    from src.utils.tf_utils import custom_signbit
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'utils')))
    from tf_utils import custom_signbit

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_signbit():
    """Test that our custom signbit function works correctly"""
    # Apply monkey patch
    if hasattr(tf.math, 'signbit'):
        original_signbit = tf.math.signbit
        tf.math.signbit = custom_signbit
        logger.info("Applied monkey patch for tf.math.signbit")
    
    try:
        # Create a tensor with some values including negative zero
        x = tf.constant([1.0, 0.0, -0.0, -1.0])
        
        # Test our patched signbit function
        result = tf.math.signbit(x)
        logger.info(f"Input tensor: {x}")
        logger.info(f"Signbit result: {result}")
        
        # Expected result: [False, False, True, True]
        expected = [False, False, True, True]
        
        # Check if results match expected
        all_match = all([r.numpy() == e for r, e in zip(result, expected)])
        if all_match:
            logger.info("✓ Test passed! Custom signbit works correctly.")
        else:
            logger.error("✗ Test failed! Results don't match expectations.")
        
        # Test the previous problematic case (using argmax with float values)
        test_float = tf.constant([0.1, 0.2, 0.3])
        _ = tf.argmax(test_float)
        logger.info("✓ Successfully ran argmax on float values without error.")
        
        return all_match
    
    except Exception as e:
        logger.error(f"Error during test: {e}")
        return False
    finally:
        # Restore original signbit if we modified it
        if 'original_signbit' in locals():
            tf.math.signbit = original_signbit
            logger.info("Restored original tf.math.signbit")

if __name__ == "__main__":
    logger.info("Testing custom signbit function")
    test_signbit()
