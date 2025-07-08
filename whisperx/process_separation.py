"""
Process separation for WhisperX to avoid PyTorch/MLX conflicts.
Runs VAD in a separate process from MLX ASR.
"""

import multiprocessing as mp
import queue
import os
import sys
from typing import List, Dict, Union, Optional, Any
import numpy as np
import pickle
import tempfile
from pathlib import Path

from whisperx.audio import load_audio, SAMPLE_RATE
from whisperx.types import TranscriptionResult


class ProcessSeparatedPipeline:
    """Pipeline that runs VAD and ASR in separate processes."""
    
    def __init__(
        self,
        asr_backend: str = "mlx",
        model_name: str = "tiny", 
        vad_method: str = "silero",
        device: str = "mlx",
        language: Optional[str] = None,
        compute_type: str = "float16",
        asr_options: Optional[Dict] = None,
        vad_options: Optional[Dict] = None,
        **kwargs
    ):
        self.asr_backend = asr_backend
        self.model_name = model_name
        self.vad_method = vad_method
        self.device = device
        self.language = language
        self.compute_type = compute_type
        self.asr_options = asr_options or {}
        self.vad_options = vad_options or {}
        self.kwargs = kwargs
        
    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        batch_size: int = 8,
        chunk_size: int = 30,
        verbose: bool = False,
        print_progress: bool = False,
    ) -> TranscriptionResult:
        """Transcribe audio using process separation."""
        
        # Load audio if path
        if isinstance(audio, str):
            audio_data = load_audio(audio)
        else:
            audio_data = audio
            
        # Step 1: Run VAD in separate process
        vad_segments = self._run_vad_process(audio_data)
        
        if verbose:
            print(f"VAD found {len(vad_segments)} segments")
        
        # Step 2: Run ASR in main process (or another process)
        segments = self._run_asr_process(
            audio_data, 
            vad_segments,
            batch_size=batch_size,
            verbose=verbose,
            print_progress=print_progress
        )
        
        return {
            "segments": segments,
            "language": self.language or "en"
        }
    
    def _run_vad_process(self, audio: np.ndarray) -> List[Dict]:
        """Run VAD in a separate process."""
        
        # Create queue for results
        result_queue = mp.Queue()
        
        # Create and start VAD process
        vad_process = mp.Process(
            target=_vad_worker,
            args=(audio, self.vad_method, self.vad_options, result_queue)
        )
        
        vad_process.start()
        
        # Wait for results with timeout
        try:
            vad_segments = result_queue.get(timeout=30)
            vad_process.join(timeout=1)
            
            if vad_process.is_alive():
                vad_process.terminate()
                
            if isinstance(vad_segments, Exception):
                raise vad_segments
                
            return vad_segments
            
        except queue.Empty:
            vad_process.terminate()
            raise TimeoutError("VAD process timed out")
    
    def _run_asr_process(
        self,
        audio: np.ndarray,
        vad_segments: List[Dict],
        batch_size: int = 8,
        verbose: bool = False,
        print_progress: bool = False,
    ) -> List[Dict]:
        """Run ASR on VAD segments."""
        
        if self.asr_backend == "mlx":
            # Can run in main process since VAD is done
            return self._run_mlx_asr(
                audio, 
                vad_segments,
                batch_size=batch_size,
                verbose=verbose,
                print_progress=print_progress
            )
        else:
            # Run in separate process for safety
            result_queue = mp.Queue()
            
            asr_process = mp.Process(
                target=_asr_worker,
                args=(
                    audio,
                    vad_segments,
                    self.asr_backend,
                    self.model_name,
                    self.language,
                    result_queue,
                    batch_size,
                    verbose,
                    print_progress
                )
            )
            
            asr_process.start()
            
            try:
                segments = result_queue.get(timeout=60)
                asr_process.join(timeout=1)
                
                if asr_process.is_alive():
                    asr_process.terminate()
                    
                if isinstance(segments, Exception):
                    raise segments
                    
                return segments
                
            except queue.Empty:
                asr_process.terminate()
                raise TimeoutError("ASR process timed out")
    
    def _run_mlx_asr(
        self,
        audio: np.ndarray,
        vad_segments: List[Dict],
        batch_size: int = 8,
        verbose: bool = False,
        print_progress: bool = False,
    ) -> List[Dict]:
        """Run MLX ASR on segments."""
        import mlx_whisper
        
        all_segments = []
        
        for i, vad_seg in enumerate(vad_segments):
            if print_progress:
                print(f"Processing segment {i+1}/{len(vad_segments)}...")
            
            # Extract audio segment
            start_sample = int(vad_seg["start"] * SAMPLE_RATE)
            end_sample = int(vad_seg["end"] * SAMPLE_RATE)
            audio_segment = audio[start_sample:end_sample]
            
            # Transcribe with MLX
            result = mlx_whisper.transcribe(
                audio_segment,
                path_or_hf_repo=f"mlx-community/whisper-{self.model_name}",
                verbose=verbose,
                language=self.language,
                temperature=0.01,
            )
            
            text = result["text"].strip()
            
            all_segments.append({
                "start": round(vad_seg["start"], 3),
                "end": round(vad_seg["end"], 3),
                "text": text
            })
        
        return all_segments


def _vad_worker(audio: np.ndarray, vad_method: str, vad_options: Dict, result_queue: mp.Queue):
    """Worker function for VAD process."""
    try:
        # Set environment for single thread
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        
        # Import here to isolate from main process
        from whisperx.vads import Silero, Pyannote
        
        # Extract VAD options
        vad_onset = vad_options.get("vad_onset", 0.5)
        vad_offset = vad_options.get("vad_offset", 0.363)
        chunk_size = vad_options.get("chunk_size", 30)
        
        # Initialize VAD
        if vad_method == "silero":
            vad = Silero(vad_onset=vad_onset, vad_offset=vad_offset, chunk_size=chunk_size)
        else:
            import torch
            device = torch.device("cpu")
            vad = Pyannote(device, use_auth_token=None, vad_onset=vad_onset)
        
        # Process audio
        waveform = vad.preprocess_audio(audio)
        vad_output = vad({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        
        # Merge chunks
        merged = vad.merge_chunks(
            vad_output,
            chunk_size=chunk_size,
            onset=vad_onset,
            offset=vad_offset
        )
        
        # Convert to serializable format
        segments = []
        for seg in merged:
            segments.append({
                "start": float(seg["start"]),
                "end": float(seg["end"])
            })
        
        result_queue.put(segments)
        
    except Exception as e:
        result_queue.put(e)


def _asr_worker(
    audio: np.ndarray,
    vad_segments: List[Dict],
    backend: str,
    model_name: str,
    language: Optional[str],
    result_queue: mp.Queue,
    batch_size: int,
    verbose: bool,
    print_progress: bool,
):
    """Worker function for ASR process."""
    try:
        # Set environment
        os.environ["OMP_NUM_THREADS"] = "1"
        
        # Import backend
        if backend == "mlx":
            import mlx_whisper
            
            segments = []
            for i, vad_seg in enumerate(vad_segments):
                if print_progress:
                    print(f"ASR {i+1}/{len(vad_segments)}")
                
                start_sample = int(vad_seg["start"] * SAMPLE_RATE)
                end_sample = int(vad_seg["end"] * SAMPLE_RATE)
                audio_segment = audio[start_sample:end_sample]
                
                result = mlx_whisper.transcribe(
                    audio_segment,
                    path_or_hf_repo=f"mlx-community/whisper-{model_name}",
                    verbose=verbose,
                    language=language,
                )
                
                segments.append({
                    "start": vad_seg["start"],
                    "end": vad_seg["end"],
                    "text": result["text"].strip()
                })
        else:
            # Use faster-whisper or other backend
            from whisperx.asr import load_model
            
            model = load_model(
                whisper_arch=model_name,
                backend=backend,
                device="cpu",
                language=language,
            )
            
            result = model.transcribe(
                audio,
                batch_size=batch_size,
                verbose=verbose,
                print_progress=print_progress,
            )
            
            segments = result["segments"]
        
        result_queue.put(segments)
        
    except Exception as e:
        result_queue.put(e)


# Alternative: Subprocess-based separation
class SubprocessPipeline:
    """Use subprocess instead of multiprocessing."""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Run transcription using subprocess."""
        import subprocess
        import json
        
        # Save config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.kwargs, f)
            config_path = f.name
        
        # Run subprocess
        cmd = [
            sys.executable,
            "-m", "whisperx.process_separation",
            "--audio", audio_path,
            "--config", config_path,
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=300
            )
            
            # Parse output
            output = json.loads(result.stdout)
            return output
            
        finally:
            Path(config_path).unlink(missing_ok=True)


def create_separated_model(
    whisper_arch: str,
    backend: str = "mlx",
    **kwargs
):
    """Create a process-separated model compatible with WhisperX API."""
    
    if backend == "mlx":
        # Use process separation for MLX
        return ProcessSeparatedPipeline(
            asr_backend=backend,
            model_name=whisper_arch,
            **kwargs
        )
    else:
        # Use regular model for other backends
        from whisperx.asr import load_model
        return load_model(whisper_arch, backend=backend, **kwargs)


if __name__ == "__main__":
    # CLI interface for subprocess mode
    import argparse
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = json.load(f)
    
    # Run pipeline
    pipeline = ProcessSeparatedPipeline(**config)
    result = pipeline.transcribe(args.audio)
    
    # Output result
    print(json.dumps(result))