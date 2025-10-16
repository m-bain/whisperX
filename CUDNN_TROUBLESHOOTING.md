# Troubleshooting cuDNN Loading Errors

If you encounter the following error when running WhisperX:

`Unable to load any of {libcudnn_cnn.so.9.1.0, libcudnn_cnn.so.9.1, libcudnn_cnn.so.9, libcudnn_cnn.so}`

This means the cuDNN libraries are installed (via whisperx dependencies) but aren't in a location where the system's dynamic linker can find them.

## Solution 1: Add to LD_LIBRARY_PATH (Recommended)

Add this at the start of your Python script or notebook:

```python
import os

# Get current LD_LIBRARY_PATH
original = os.environ.get("LD_LIBRARY_PATH", "")

cudnn_path = "/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib/"
os.environ['LD_LIBRARY_PATH'] = original + ":" + cudnn_path
```

**Note:** Adjust the Python version (`python3.12`) to match your environment.

## Solution 2: Symlink to LD_LIBRARY_PATH Directory

If Solution 1 didn't work and you still get the "unable to load" error, symlink the libraries to a directory that's already in your `LD_LIBRARY_PATH`:

1. Check what's in your LD_LIBRARY_PATH: `echo "$LD_LIBRARY_PATH"`
2. Assuming that there is only one path set.  
   Symlink the downloaded libcudnn files to that path:  
   `ln -s /usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib/libcudnn* "$LD_LIBRARY_PATH"/`

   **Note:** If `LD_LIBRARY_PATH` contains multiple paths (separated by `:`), pick one directory and use it instead of `"$LD_LIBRARY_PATH"`. For example: `/usr/lib/x86_64-linux-gnu/`

