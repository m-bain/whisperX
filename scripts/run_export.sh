#!/bin/bash

# Usage: ./launch_script.sh --data-root /path/to/data --save-root /path/to/save --compute-type float32 --file-type wav --skip-existing

export WORLD_SIZE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) # Count available GPUs

# Check if GPUs are available
if [ "$WORLD_SIZE" -lt 1 ]; then
    echo "Error: No GPUs found for distributed processing."
    exit 1
fi

# Activate virtual environment
eval "$(conda shell.bash hook)"
conda activate whisperx

# Run the Python script using torch.multiprocessing spawn
python -m ../whisperx/prosody_features/extract_prosody_features.py "$@"

conda deactivate
