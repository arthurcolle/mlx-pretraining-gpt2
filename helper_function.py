"""
Helper function to provide compatibility with older and newer MLX versions.
This addresses the issue where mx.simplify() was removed in newer versions of MLX.
"""

import mlx.core as mx
import inspect

def simplify_compatible(*args):
    """
    Provides compatibility for mx.simplify across different MLX versions.
    
    In MLX 0.0.10, simplify() was available and used to optimize computation graphs.
    In newer versions, this function was removed.
    
    This helper function checks if simplify exists in the current MLX version and
    calls it if available; otherwise, it does nothing (silently returns).
    
    Args:
        *args: Any number of arrays and/or trees of arrays to be simplified.
    """
    if hasattr(mx, 'simplify') and callable(getattr(mx, 'simplify')):
        # MLX 0.0.10 has the simplify function
        mx.simplify(*args)
    else:
        # Newer versions of MLX don't have simplify
        # The computation graph optimization is likely handled automatically
        pass