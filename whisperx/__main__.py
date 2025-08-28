import argparse
import importlib.metadata
import platform

import torch

from whisperx.utils import (LANGUAGES, TO_LANGUAGE_CODE, optional_float,
                            optional_int, str2bool)
from whisperx.hardware import assess_optimal_settings, detect_gpu_capabilities, detect_cuda_environment


def cli():
    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio", nargs="+", type=str, help="audio file(s) to transcribe")
    parser.add_argument("--model", default="small", help="name of the Whisper model to use")
    parser.add_argument("--model_cache_only", type=str2bool, default=False, help="If True, will not attempt to download models, instead using cached models from --model_dir")
    parser.add_argument("--model_dir", type=str, default=None, help="the path to save model files; uses ~/.cache/whisper by default")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="device to use for PyTorch inference (auto=detect best available)")
    parser.add_argument("--device_index", default=0, type=int, help="device index to use for FasterWhisper inference")
    parser.add_argument("--batch_size", default=8, type=int, help="the preferred batch size for inference")
    parser.add_argument("--compute_type", default="auto", choices=["auto", "float16", "float32", "int8"],
                       help="compute type for computation (auto=detect optimal)")

    parser.add_argument("--no_hardware_detection", type=str2bool, default=False,
                       help="disable automatic hardware detection and optimization")
    parser.add_argument("--show_hardware_info", type=str2bool, default=False,
                       help="display detailed hardware information and exit")


    parser.add_argument("--output_dir", "-o", type=str, default=".", help="directory to save the outputs")
    parser.add_argument("--output_format", "-f", type=str, default="all", choices=["all", "srt", "vtt", "txt", "tsv", "json", "aud"], help="format of the output file; if not specified, all available formats will be produced")
    parser.add_argument("--verbose", type=str2bool, default=True, help="whether to print out the progress and debug messages")

    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default=None, choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]), help="language spoken in the audio, specify None to perform language detection")

    # alignment params
    parser.add_argument("--align_model", default=None, help="Name of phoneme-level ASR model to do alignment")
    parser.add_argument("--interpolate_method", default="nearest", choices=["nearest", "linear", "ignore"], help="For word .srt, method to assign timestamps to non-aligned words, or merge them into neighbouring.")
    parser.add_argument("--no_align", action='store_true', help="Do not perform phoneme alignment")
    parser.add_argument("--return_char_alignments", action='store_true', help="Return character-level alignments in the output json file")

    # vad params
    parser.add_argument("--vad_method", type=str, default="pyannote", choices=["pyannote", "silero"], help="VAD method to be used")
    parser.add_argument("--vad_onset", type=float, default=0.500, help="Onset threshold for VAD (see pyannote.audio), reduce this if speech is not being detected")
    parser.add_argument("--vad_offset", type=float, default=0.363, help="Offset threshold for VAD (see pyannote.audio), reduce this if speech is not being detected.")
    parser.add_argument("--chunk_size", type=int, default=30, help="Chunk size for merging VAD segments. Default is 30, reduce this if the chunk is too long.")

    # diarization params
    parser.add_argument("--diarize", action="store_true", help="Apply diarization to assign speaker labels to each segment/word")
    parser.add_argument("--min_speakers", default=None, type=int, help="Minimum number of speakers to in audio file")
    parser.add_argument("--max_speakers", default=None, type=int, help="Maximum number of speakers to in audio file")
    parser.add_argument("--diarize_model", default="pyannote/speaker-diarization-3.1", type=str, help="Name of the speaker diarization model to use")
    parser.add_argument("--speaker_embeddings", action="store_true", help="Include speaker embeddings in JSON output (only works with --diarize)")

    parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
    parser.add_argument("--best_of", type=optional_int, default=5, help="number of candidates when sampling with non-zero temperature")
    parser.add_argument("--beam_size", type=optional_int, default=5, help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=1.0, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

    parser.add_argument("--suppress_tokens", type=str, default="-1", help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    parser.add_argument("--suppress_numerals", action="store_true", help="whether to suppress numeric symbols and currency symbols during sampling, since wav2vec2 cannot align them correctly")

    parser.add_argument("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
    parser.add_argument("--condition_on_previous_text", type=str2bool, default=False, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")
    parser.add_argument("--fp16", type=str2bool, default=True, help="whether to perform inference in fp16; True by default")

    parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")

    parser.add_argument("--max_line_width", type=optional_int, default=None, help="(not possible with --no_align) the maximum number of characters in a line before breaking the line")
    parser.add_argument("--max_line_count", type=optional_int, default=None, help="(not possible with --no_align) the maximum number of lines in a segment")
    parser.add_argument("--highlight_words", type=str2bool, default=False, help="(not possible with --no_align) underline each word as it is spoken in srt and vtt")
    parser.add_argument("--segment_resolution", type=str, default="sentence", choices=["sentence", "chunk"], help="(not possible with --no_align) the maximum number of characters in a line before breaking the line")

    parser.add_argument("--threads", type=optional_int, default=0, help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")

    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face Access Token to access PyAnnote gated models")

    parser.add_argument("--print_progress", type=str2bool, default = False, help = "if True, progress will be printed in transcribe() and align() methods.")
    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {importlib.metadata.version('whisperx')}",help="Show whisperx version information and exit")
    parser.add_argument("--python-version", "-P", action="version", version=f"Python {platform.python_version()} ({platform.python_implementation()})",help="Show python version information and exit")
    # fmt: on

    args = parser.parse_args().__dict__

    if args.pop("show_hardware_info"):
        show_hardware_info()
        return

    if not args.pop("no_hardware_detection"):
        args = apply_hardware_optimization(args)

    from whisperx.transcribe import transcribe_task

    transcribe_task(args, parser)


def show_hardware_info():
    """Display comprehensive hardware information."""
    print("=== WhisperX Hardware Information ===\n")

    try:
        # Detect hardware
        gpus = detect_gpu_capabilities()
        cuda_env = detect_cuda_environment()
        hardware_caps = assess_optimal_settings(gpus, cuda_env)

        # Display system info
        print(f"System: {platform.system()} {platform.release()}")
        print(f"CPU Cores: {hardware_caps.cpu_cores}")
        print(f"System RAM: {hardware_caps.system_ram} MB")
        print()

        # Display CUDA info
        print("CUDA Environment:")
        print(f"  CUDA Available: {cuda_env.cuda_available}")
        if cuda_env.cuda_available:
            print(f"  PyTorch CUDA Version: {cuda_env.pytorch_cuda_version}")
            print(f"  System CUDA Version: {cuda_env.cuda_version or 'Not detected'}")
            print(f"  cuDNN Version: {cuda_env.cudnn_version or 'Not detected'}")
            print(f"  Compatible: {cuda_env.is_compatible}")
            if cuda_env.compatibility_issues:
                print(f"  Issues: {', '.join(cuda_env.compatibility_issues)}")
        print()

        # Display GPU info
        if gpus:
            print("Detected GPUs:")
            for gpu in gpus:
                print(f"  GPU {gpu.index}: {gpu.name}")
                print(f"    VRAM: {gpu.vram_total} MB total, {gpu.vram_free} MB free")
                print(f"    Compute Capability: {gpu.compute_capability[0]}.{gpu.compute_capability[1]}")
                if gpu.cuda_cores:
                    print(f"    CUDA Cores: {gpu.cuda_cores}")
        else:
            print("No GPUs detected")
        print()

        # Display recommendations
        print("Recommended Settings:")
        print(f"  Device: {hardware_caps.recommended_device}")
        if hardware_caps.recommended_device == "cuda":
            print(f"  Device Index: {hardware_caps.recommended_device_index}")
        print(f"  Compute Type: {hardware_caps.recommended_compute_type}")
        print(f"  Batch Size: {hardware_caps.recommended_batch_size} (max: {hardware_caps.max_batch_size})")
        print(f"  Expected Speedup: {hardware_caps.estimated_speedup:.1f}x")

        if hardware_caps.warning_messages:
            print("\nWarnings:")
            for warning in hardware_caps.warning_messages:
                print(f"  - {warning}")

    except Exception as e:
        print(f"Error detecting hardware: {e}")


def apply_hardware_optimization(args: dict) -> dict:
    """Apply hardware-based optimization to arguments."""
    try:
        # Get model size for optimization
        model_size = args.get("model", "small")

        # Detect hardware capabilities
        gpus = detect_gpu_capabilities()
        cuda_env = detect_cuda_environment()
        hardware_caps = assess_optimal_settings(gpus, cuda_env, model_size)

        # Apply auto device selection
        if args.get("device") == "auto":
            args["device"] = hardware_caps.recommended_device
            args["device_index"] = hardware_caps.recommended_device_index
            print(f">>Auto-detected device: {args['device']}")

        # Apply auto compute type selection
        if args.get("compute_type") == "auto":
            args["compute_type"] = hardware_caps.recommended_compute_type
            print(f">>Auto-detected compute type: {args['compute_type']}")

        # Apply auto batch size selection
        if args.get("batch_size") == "auto":
            args["batch_size"] = hardware_caps.recommended_batch_size
            print(f">>Auto-detected batch size: {args['batch_size']}")
        elif isinstance(args.get("batch_size"), str) and args["batch_size"].isdigit():
            args["batch_size"] = int(args["batch_size"])

        # Display hardware summary
        if hardware_caps.can_use_gpu:
            best_gpu = max(hardware_caps.gpus, key=lambda g: g.vram_total)
            print(f">>Using GPU: {best_gpu.name} ({best_gpu.vram_total}MB VRAM)")
        else:
            print(">>Using CPU processing")

        # Display warnings
        if hardware_caps.warning_messages:
            for warning in hardware_caps.warning_messages:
                print(f">>Warning: {warning}")

    except Exception as e:
        print(f">>Hardware detection failed: {e}, using default settings")
        # Fallback to basic detection
        if args.get("device") == "auto":
            args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        if args.get("compute_type") == "auto":
            args["compute_type"] = "float16" if torch.cuda.is_available() else "int8"
        if args.get("batch_size") == "auto":
            args["batch_size"] = 8

    return args

if __name__ == "__main__":
    cli()
