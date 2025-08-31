#!/usr/bin/env python3
# test_fix.py - A simple script to test that our TensorFlow monkey patch works correctly

import os
import numpy as np
import tensorflow as tf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_tensorflow_argmax():
    """Test that TensorFlow argmax works correctly with float values (the original issue)"""
    try:
        # Test the previous problematic case (using argmax with float values)
        test_float = tf.constant([0.1, 0.2, 0.3])
        result = tf.argmax(test_float)
        logger.info(f"Test input: {test_float}")
        logger.info(f"Argmax result: {result}")

        # Verify the result is correct (should return index 2, the highest value)
        expected_index = 2
        assert result.numpy() == expected_index, f"Expected argmax to return {expected_index}, got {result.numpy()}"
        logger.info("✓ Test passed! TensorFlow argmax works correctly with float values.")

        # Test with more complex float array
        test_float2 = tf.constant([0.5, 0.8, 0.3, 0.9, 0.1])
        result2 = tf.argmax(test_float2)
        expected_index2 = 3  # 0.9 is at index 3
        assert result2.numpy() == expected_index2, f"Expected argmax to return {expected_index2}, got {result2.numpy()}"
        logger.info("✓ Test passed! TensorFlow argmax works correctly with complex float arrays.")

    except Exception as e:
        logger.error(f"Error during test: {e}")
        assert False, f"Error during test: {e}"

if __name__ == "__main__":
    logger.info("Testing TensorFlow argmax functionality")
    test_tensorflow_argmax()
