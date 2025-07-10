"""
Patch for pyannote model loading compatibility with PyTorch 2.6+

This module provides a fix for the weights_only issue when loading
pyannote VAD models in PyTorch 2.6 and later.
"""

import torch
import warnings
from functools import wraps

# Store the original torch.load function
_original_torch_load = torch.load

def patch_torch_load_for_pyannote():
    """Apply a patch to torch.load to handle pyannote model loading issues."""
    
    @wraps(_original_torch_load)
    def patched_load(f, map_location=None, pickle_module=None, *, weights_only=None, **pickle_load_args):
        """
        Patched torch.load that handles pyannote compatibility.
        
        This specifically addresses the weights_only default change in PyTorch 2.6+
        for pyannote models which require weights_only=False to load properly.
        """
        # If weights_only is explicitly set, respect it
        if weights_only is not None:
            return _original_torch_load(
                f, map_location=map_location, 
                pickle_module=pickle_module,
                weights_only=weights_only,
                **pickle_load_args
            )
        
        # Check if this is being called from pyannote or pytorch-lightning context
        import inspect
        stack = inspect.stack()
        
        # Look for pyannote or lightning in the call stack
        is_pyannote_context = any(
            'pyannote' in str(frame.filename) or 
            'lightning' in str(frame.filename) or
            'whisperx/vad' in str(frame.filename)
            for frame in stack[:15]
        )
        
        if is_pyannote_context:
            # For pyannote models, we need weights_only=False
            warnings.warn(
                "Loading model with weights_only=False for pyannote compatibility. "
                "This is less secure but required for pyannote VAD models.",
                UserWarning,
                stacklevel=2
            )
            weights_only = False
        
        # Use the original function with our determined weights_only value
        return _original_torch_load(
            f, map_location=map_location,
            pickle_module=pickle_module,
            weights_only=weights_only,
            **pickle_load_args
        )
    
    # Apply the patch
    torch.load = patched_load
    
    # Also patch the pytorch-lightning load function if available
    try:
        import lightning_fabric.utilities.cloud_io as lightning_io
        if hasattr(lightning_io, '_load'):
            original_pl_load = lightning_io._load
            
            @wraps(original_pl_load)
            def patched_pl_load(path_or_url, map_location=None):
                # Force weights_only=False for lightning loads
                return torch.load(path_or_url, map_location=map_location, weights_only=False)
            
            lightning_io._load = patched_pl_load
    except ImportError:
        pass

# Auto-apply the patch when this module is imported
patch_torch_load_for_pyannote()