import tensorflow as tf
import numpy as np

def custom_signbit(x):
    """Custom implementation of signbit that avoids the OverflowError."""
    x = tf.convert_to_tensor(x)
    dtype = x.dtype
    if dtype in [tf.int32, tf.int64, tf.uint8, tf.uint16, tf.uint32, tf.uint64]:
        return tf.less(x, 0)
    elif dtype == tf.bool:
        return tf.fill(tf.shape(x), False)
    else:
        # For float types, we use a trick that avoids the 0x80000000 constant
        # that causes OverflowError in some systems
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
