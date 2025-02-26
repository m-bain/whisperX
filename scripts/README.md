## Running the Export Script

To run the export script, use the `run_export.sh` script located in the `scripts` directory. This script sets up the environment and runs the feature extraction process using multiple GPUs.

### Usage

```sh
run_export.sh --data-root /path/to/data --save-root /path/to/save --compute-type float32 --file-type wav --skip-existing
```

#### Arguments

--data-root: Root data directory containing the audio files.  
--save-root: Root directory for saving the extracted prosody features. Directory structure will be  automatically be created to mirror the audio file directories.  
--compute-type: Compute format type (default: float32).   
--file-type: Type of audio file to process (default: wav).  
--skip-existing: Skip processing of existing files in save directory.