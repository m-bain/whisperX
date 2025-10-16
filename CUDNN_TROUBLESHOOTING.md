# Troubleshooting cuDNN Loading Errors

This guide helps resolve common cuDNN-related errors when running WhisperX on GPU. These issues typically occur when the system can't locate cuDNN libraries or finds conflicting versions.

## Unable to Load cuDNN Libraries

If you encounter the following error when running WhisperX:

`Unable to load any of {libcudnn_cnn.so.9.1.0, libcudnn_cnn.so.9.1, libcudnn_cnn.so.9, libcudnn_cnn.so}`

This means the cuDNN libraries are installed (via whisperx dependencies) but aren't in a location where the system's dynamic linker can find them.

### Solution 1: Add to LD_LIBRARY_PATH (Recommended)

Add this at the start of your Python script or notebook:

```python
import os

# Get current LD_LIBRARY_PATH
original = os.environ.get("LD_LIBRARY_PATH", "")

cudnn_path = "/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib/"
os.environ['LD_LIBRARY_PATH'] = original + ":" + cudnn_path
```

**Note:** Adjust the Python version (`python3.12`) to match your environment.

### Solution 2: Symlink to LD_LIBRARY_PATH Directory

If Solution 1 didn't work and you still get the "unable to load" error, symlink the libraries to a directory that's already in your `LD_LIBRARY_PATH`:

1. Check what's in your LD_LIBRARY_PATH: `echo "$LD_LIBRARY_PATH"`
2. Assuming that there is only one path set.  
   Symlink the downloaded libcudnn files to that path:  
   `ln -s /usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib/libcudnn* "$LD_LIBRARY_PATH"/`

   **Note:** If `LD_LIBRARY_PATH` contains multiple paths (separated by `:`), pick one directory and use it instead of `"$LD_LIBRARY_PATH"`. For example: `/usr/lib/x86_64-linux-gnu/`

## cuDNN Version Incompatibility

If you encounter this error:

```
RuntimeError: cuDNN version incompatibility: PyTorch was compiled against (9, 10, 2) but found runtime version (9, 2, 1)
```

This means PyTorch is finding a different cuDNN version than the one it was compiled with. **PyTorch comes bundled with its own cuDNN**, but a conflicting cuDNN in `LD_LIBRARY_PATH` is taking precedence.

### Solution: Remove Conflicting cuDNN from Path

Check if there's a conflicting cuDNN path:

```bash
echo $LD_LIBRARY_PATH
```

If you see paths pointing to older cuDNN installations (e.g., system-installed cuDNN or manually downloaded), try one of these:

**Option 1: Clear LD_LIBRARY_PATH temporarily**

```python
import os
# Let PyTorch use its bundled cuDNN
os.environ.pop('LD_LIBRARY_PATH', None)
```

**Option 2: Set LD_LIBRARY_PATH to only the correct version**

```python
import os
# Point only to the cuDNN that matches PyTorch's compiled version
os.environ['LD_LIBRARY_PATH'] = "/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib/"
```

**Note:** This error is unlikely on a clean install. If it occurs anyway, [open an issue](https://github.com/m-bain/whisperX/issues). If you've modified system libraries or CUDA/cuDNN, the options above should help resolve most cases.
