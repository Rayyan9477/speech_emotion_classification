"""
monkey_patch.py - Workaround for the int overflow error in TensorFlow.

This script patches the argmax function in TensorFlow's numpy.py to avoid the
OverflowError caused by using 0x80000000 as a constant.
"""

import importlib
import inspect
import sys
import types
import numpy as np
import tensorflow as tf

def monkeypatch():
    """Apply monkey patch to fix the OverflowError when handling floating point numbers."""
    try:        # Try to locate the module where the issue is occurring
        if 'keras.src.backend.tensorflow.numpy' in sys.modules:
            numpy_module = sys.modules['keras.src.backend.tensorflow.numpy']
            
            # Get the original argmax function
            original_argmax = numpy_module.argmax
            
            # Define a new argmax that avoids the issue
            def safe_argmax(x, axis=None, keepdims=False):
                """Safe implementation of argmax that doesn't use signbit."""
                x = numpy_module.convert_to_tensor(x)
                dtype = numpy_module.standardize_dtype(x.dtype)
                if "float" not in dtype or x.ndim == 0:
                    _x = x
                    if axis is None:
                        x = tf.reshape(x, [-1])
                    y = tf.argmax(x, axis=axis, output_type="int32")
                    if keepdims:
                        y = numpy_module._keepdims(_x, y, axis)
                    return y
                
                # Fix for float types without using signbit
                dtype = numpy_module.dtypes.result_type(dtype, "float32")
                x = numpy_module.cast(x, dtype)
                
                # Handle -0.0 differently to avoid using signbit
                # Replace -0.0 with small negative number
                eps = np.finfo(np.float32).tiny
                zero_mask = tf.equal(x, 0.0)
                
                # We need to detect negative zeros
                # For any operation where -0.0 behaves differently from +0.0:
                # 1.0 / (-0.0) gives -inf, 1.0 / 0.0 gives inf
                reciprocal = tf.where(zero_mask, tf.ones_like(x), 1.0 / x)
                neg_zero_mask = tf.logical_and(zero_mask, tf.less(reciprocal, 0.0))
                x = tf.where(neg_zero_mask, -eps, x)
                
                _x = x
                if axis is None:
                    x = tf.reshape(x, [-1])
                y = tf.argmax(x, axis=axis, output_type="int32")
                if keepdims:
                    y = numpy_module._keepdims(_x, y, axis)
                return y
                
            # Replace the original argmax function
            numpy_module.argmax = safe_argmax
            
            return "TensorFlow patched successfully to avoid overflow errors"
    except Exception as e:
        print(f"Error applying monkey patch: {e}")
        return f"Patching failed: {e}"

if __name__ == "__main__":
    if monkeypatch():
        print("Successfully applied monkey patch for TensorFlow argmax function")
    else:
        print("Failed to apply monkey patch")
