#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config_path>"
    exit 1
fi

# Assign the first argument to a variable
CONFIG_PATH=$1

# Activate virtual environment
eval "$(conda shell.bash hook)"
conda activate arts-prosody

# Run the Python script with the provided configuration path
python3 train_prosody_model.py "$CONFIG_PATH"