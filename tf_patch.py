"""
Patch for TensorFlow numpy.py to fix the 0x80000000 OverflowError.
This file should be imported before running any TensorFlow operations.
"""

import importlib
import sys
import tensorflow as tf
import types
import warnings

def fix_tf_signbit_overflow():
    """
    Fix for the OverflowError caused by using 0x80000000 as a constant,
    which is too large for a C long on some platforms.
    """
    try:
        # First, let's check if the problematic function exists
        if hasattr(tf.math, 'signbit') and tf.math.signbit.__module__ == 'tensorflow._api.v2.math':
            # Define our custom signbit function
            def custom_signbit(x):
                """Custom implementation of signbit that avoids the OverflowError."""
                x = tf.convert_to_tensor(x)
                dtype = x.dtype
                
                # Handle integer and boolean types
                if dtype in (tf.int32, tf.int64, tf.uint8, tf.uint16, tf.uint32, tf.uint64):
                    return tf.less(x, 0)
                elif dtype == tf.bool:
                    return tf.fill(tf.shape(x), False)
                else:
                    # For float types, use a different approach to avoid the 0x80000000 constant
                    zero = tf.constant(0.0, dtype=dtype)
                    # First check if x equals zero
                    is_zero = tf.equal(x, zero)
                    # Then check for negative zero specifically (-0.0 == 0.0 but 1/(-0.0) < 0)
                    # We need to handle potential division by zero
                    safe_recip = tf.where(is_zero, tf.ones_like(x), 1.0/x)
                    is_negative_zero = tf.logical_and(is_zero, tf.less(safe_recip, zero))
                    # Handle normal negative numbers
                    is_negative = tf.less(x, zero)
                    return tf.logical_or(is_negative, is_negative_zero)
            
            # Replace the original function with our custom implementation
            tf.math.signbit = custom_signbit
            return True
        
        # Alternative approach: try to patch at a lower level
        try:
            # Get the keras backend numpy module
            keras_numpy_module = importlib.import_module('keras.src.backend.tensorflow.numpy')
            
            # Check if the module contains the signbit function
            if hasattr(keras_numpy_module, 'signbit'):
                original_signbit = keras_numpy_module.signbit
                
                # Define a wrapper that replaces the problematic part
                def signbit_wrapper(x):
                    """Wrapper for the signbit function to avoid the OverflowError."""
                    # Implementation similar to our custom function
                    x = tf.convert_to_tensor(x)
                    dtype = x.dtype
                    
                    # Handle integer and boolean types
                    if tf.as_dtype(dtype).is_integer or dtype == tf.bool:
                        return tf.less(x, 0)
                    else:
                        # For float types, use a different approach
                        zero = tf.constant(0.0, dtype=dtype)
                        is_zero = tf.equal(x, zero)
                        safe_recip = tf.where(is_zero, tf.ones_like(x), 1.0/x)
                        is_negative_zero = tf.logical_and(is_zero, tf.less(safe_recip, zero))
                        is_negative = tf.less(x, zero)
                        return tf.logical_or(is_negative, is_negative_zero)
                
                # Replace the function
                keras_numpy_module.signbit = signbit_wrapper
                return True
        except (ImportError, AttributeError) as e:
            warnings.warn(f"Failed to patch keras numpy module: {e}")
            return False
        
        return False
    except Exception as e:
        warnings.warn(f"Failed to apply TensorFlow signbit patch: {e}")
        return False

# Apply the patch
patched = fix_tf_signbit_overflow()
if patched:
    print("Successfully patched TensorFlow signbit function to avoid OverflowError")
else:
    print("Failed to patch TensorFlow signbit function")
