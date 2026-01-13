import argparse
import gc
import os
import warnings

import numpy as np
import torch

from whisperx.alignment import align, load_align_model
from whisperx.asr import load_model
from whisperx.audio import load_audio
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
from whisperx.schema import AlignedTranscriptionResult, TranscriptionResult
from whisperx.utils import LANGUAGES, TO_LANGUAGE_CODE, get_writer
from typing import Callable, Optional
from whisperx.log_utils import get_logger

logger = get_logger(__name__)


def transcribe_task(args: dict, parser: argparse.ArgumentParser):
    """Transcription task to be called from CLI.

    Args:
        args: Dictionary of command-line arguments.
        parser: argparse.ArgumentParser object.
    """
    # fmt: off

    model_name: str = args.pop("model")
    batch_size: int = args.pop("batch_size")
    model_dir: str = args.pop("model_dir")
    model_cache_only: bool = args.pop("model_cache_only")
    output_dir: str = args.pop("output_dir")
    output_format: str = args.pop("output_format")
    device: str = args.pop("device")
    device_index: int = args.pop("device_index")
    compute_type: str = args.pop("compute_type")
    verbose: bool = args.pop("verbose")

    # model_flush: bool = args.pop("model_flush")
    os.makedirs(output_dir, exist_ok=True)

    align_model: str = args.pop("align_model")
    interpolate_method: str = args.pop("interpolate_method")
    no_align: bool = args.pop("no_align")
    task: str = args.pop("task")
    if task == "translate":
        # translation cannot be aligned
        no_align = True

    return_char_alignments: bool = args.pop("return_char_alignments")

    hf_token: str = args.pop("hf_token")
    vad_method: str = args.pop("vad_method")
    vad_onset: float = args.pop("vad_onset")
    vad_offset: float = args.pop("vad_offset")

    chunk_size: int = args.pop("chunk_size")

    diarize: bool = args.pop("diarize")
    min_speakers: int = args.pop("min_speakers")
    max_speakers: int = args.pop("max_speakers")
    diarize_model_name: str = args.pop("diarize_model")
    print_progress: bool = args.pop("print_progress")
    return_speaker_embeddings: bool = args.pop("speaker_embeddings")

    if return_speaker_embeddings and not diarize:
        warnings.warn("--speaker_embeddings has no effect without --diarize")

    if args["language"] is not None:
        args["language"] = args["language"].lower()
        if args["language"] not in LANGUAGES:
            if args["language"] in TO_LANGUAGE_CODE:
                args["language"] = TO_LANGUAGE_CODE[args["language"]]
            else:
                raise ValueError(f"Unsupported language: {args['language']}")

    if model_name.endswith(".en") and args["language"] != "en":
        if args["language"] is not None:
            warnings.warn(
                f"{model_name} is an English-only model but received '{args['language']}'; using English instead."
            )
        args["language"] = "en"
    align_language = (
        args["language"] if args["language"] is not None else "en"
    )  # default to loading english if not specified

    temperature = args.pop("temperature")
    if (increment := args.pop("temperature_increment_on_fallback")) is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = [temperature]

    faster_whisper_threads = 4
    if (threads := args.pop("threads")) > 0:
        torch.set_num_threads(threads)
        faster_whisper_threads = threads

    asr_options = {
        "beam_size": args.pop("beam_size"),
        "patience": args.pop("patience"),
        "length_penalty": args.pop("length_penalty"),
        "temperatures": temperature,
        "compression_ratio_threshold": args.pop("compression_ratio_threshold"),
        "log_prob_threshold": args.pop("logprob_threshold"),
        "no_speech_threshold": args.pop("no_speech_threshold"),
        "condition_on_previous_text": False,
        "initial_prompt": args.pop("initial_prompt"),
        "hotwords": args.pop("hotwords"),
        "suppress_tokens": [int(x) for x in args.pop("suppress_tokens").split(",")],
        "suppress_numerals": args.pop("suppress_numerals"),
    }

    writer = get_writer(output_format, output_dir)
    word_options = ["highlight_words", "max_line_count", "max_line_width"]
    if no_align:
        for option in word_options:
            if args[option]:
                parser.error(f"--{option} not possible with --no_align")
    if args["max_line_count"] and not args["max_line_width"]:
        warnings.warn("--max_line_count has no effect without --max_line_width")
    writer_args = {arg: args.pop(arg) for arg in word_options}

    # Part 1: VAD & ASR Loop
    results = []
    # model = load_model(model_name, device=device, download_root=model_dir)
    model = load_model(
        model_name,
        device=device,
        device_index=device_index,
        download_root=model_dir,
        compute_type=compute_type,
        language=args["language"],
        asr_options=asr_options,
        vad_method=vad_method,
        vad_options={
            "chunk_size": chunk_size,
            "vad_onset": vad_onset,
            "vad_offset": vad_offset,
        },
        task=task,
        local_files_only=model_cache_only,
        threads=faster_whisper_threads,
    )

    for audio_path in args.pop("audio"):
        audio = load_audio(audio_path)
        # >> VAD & ASR
        logger.info("Performing transcription...")
        result: TranscriptionResult = model.transcribe(
            audio,
            batch_size=batch_size,
            chunk_size=chunk_size,
            print_progress=print_progress,
            verbose=verbose,
        )
        results.append((result, audio_path))

    # Unload Whisper and VAD
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Part 2: Align Loop
    if not no_align:
        tmp_results = results
        results = []
        align_model, align_metadata = load_align_model(
            align_language, device, model_name=align_model
        )
        for result, audio_path in tmp_results:
            # >> Align
            if len(tmp_results) > 1:
                input_audio = audio_path
            else:
                # lazily load audio from part 1
                input_audio = audio

            if align_model is not None and len(result["segments"]) > 0:
                if result.get("language", "en") != align_metadata["language"]:
                    # load new language
                    logger.info(
                        f"New language found ({result['language']})! Previous was ({align_metadata['language']}), loading new alignment model for new language..."
                    )
                    align_model, align_metadata = load_align_model(
                        result["language"], device
                    )
                logger.info("Performing alignment...")
                result: AlignedTranscriptionResult = align(
                    result["segments"],
                    align_model,
                    align_metadata,
                    input_audio,
                    device,
                    interpolate_method=interpolate_method,
                    return_char_alignments=return_char_alignments,
                    print_progress=print_progress,
                )

            results.append((result, audio_path))

        # Unload align model
        del align_model
        gc.collect()
        torch.cuda.empty_cache()

    # >> Diarize
    if diarize:
        if hf_token is None:
            logger.warning(
                "No --hf_token provided, needs to be saved in environment variable, otherwise will throw error loading diarization model"
            )
        tmp_results = results
        logger.info("Performing diarization...")
        logger.info(f"Using model: {diarize_model_name}")
        results = []
        diarize_model = DiarizationPipeline(model_name=diarize_model_name, use_auth_token=hf_token, device=device)
        for result, input_audio_path in tmp_results:
            diarize_result = diarize_model(
                input_audio_path, 
                min_speakers=min_speakers, 
                max_speakers=max_speakers, 
                return_embeddings=return_speaker_embeddings
            )

            if return_speaker_embeddings:
                diarize_segments, speaker_embeddings = diarize_result
            else:
                diarize_segments = diarize_result
                speaker_embeddings = None

            result = assign_word_speakers(diarize_segments, result, speaker_embeddings)
            results.append((result, input_audio_path))
    # >> Write
    for result, audio_path in results:
        result["language"] = align_language
        writer(result, audio_path, writer_args)


def transcribe_with_callbacks(
        audio_file: str,
        model_name: str = "base",
        device: str = "auto",
        language: Optional[str] = None,
        enable_alignment: bool = True,
        enable_diarization: bool = False,
        progress_callback: Optional[Callable[[int], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        cached_models: Optional[dict] = None,
        return_models: bool = False,
        **kwargs
) -> dict:
    """
    Transcribe audio with progress callbacks for GUI integration.
    Uses actual progress values from WhisperX instead of hardcoded values.

    Args:
        audio_file: Path to audio file
        model_name: Name of Whisper model to use
        device: Device to use (cuda, cpu, auto)
        language: Language code (None for auto-detection)
        enable_alignment: Enable word-level alignment
        enable_diarization: Enable speaker diarization
        progress_callback: Callback for progress updates (0-100)
        status_callback: Callback for status messages
        cached_models: Optional dict of pre-loaded models {'asr': model, 'alignment': {...}, 'diarization': pipeline}
        return_models: If True, return loaded models in result for caching
        **kwargs: Additional parameters (device_index, compute_type, batch_size, hf_token)

    Returns:
        Dictionary containing transcription results and optionally loaded models
    """

    # Progress phase allocation
    phase_weights = {
        'loading': 10,
        'transcription': 50,
        'alignment': 25,
        'diarization': 15
    }

    current_phase_start = 0
    loaded_models = {}

    def phase_progress_callback(phase: str, progress: int):
        """Convert phase-specific progress to overall progress."""
        if progress_callback:
            phase_weight = phase_weights[phase]
            overall_progress = current_phase_start + int((progress * phase_weight) / 100)
            progress_callback(min(overall_progress, 100))

    try:
        # Phase 1: Loading
        if status_callback:
            status_callback("Loading audio...")
        phase_progress_callback('loading', 20)

        audio = load_audio(audio_file)
        phase_progress_callback('loading', 50)

        # Use cached ASR model or load new one
        if cached_models and 'asr' in cached_models:
            if status_callback:
                status_callback("Using cached ASR model...")
            model = cached_models['asr']
        else:
            if status_callback:
                status_callback("Loading ASR model...")
            model = load_model(
                whisper_arch=model_name,
                device=device,
                language=language,
                **{k: v for k, v in kwargs.items() if k in ['device_index', 'compute_type', 'batch_size']}
            )
            loaded_models['asr'] = model

        phase_progress_callback('loading', 100)
        current_phase_start += phase_weights['loading']

        # Phase 2: Transcription with real progress
        if status_callback:
            status_callback("Transcribing audio...")

        result = model.transcribe(
            audio,
            batch_size=kwargs.get('batch_size', 8),
            language=language,
            print_progress=False,  # Disable printing, use callback
            combined_progress=False,  # We handle phase combination
            progress_callback=lambda p: phase_progress_callback('transcription', p),
            status_callback=status_callback
        )

        current_phase_start += phase_weights['transcription']
        final_result = {"transcription": result}

        # Phase 3: Alignment with real progress
        if enable_alignment:
            if status_callback:
                status_callback("Aligning timestamps...")

            # Use cached alignment model or load new one
            align_language = language or "en"
            if cached_models and 'alignment' in cached_models:
                cached_align_lang = cached_models['alignment'].get('metadata', {}).get('language', 'en')
                if cached_align_lang == align_language:
                    if status_callback:
                        status_callback("Using cached alignment model...")
                    model_a = cached_models['alignment']['model']
                    metadata = cached_models['alignment']['metadata']
                else:
                    if status_callback:
                        status_callback("Loading alignment model for new language...")
                    model_a, metadata = load_align_model(
                        language_code=align_language,
                        device=device
                    )
                    loaded_models['alignment'] = {'model': model_a, 'metadata': metadata}
            else:
                model_a, metadata = load_align_model(
                    language_code=align_language,
                    device=device
                )
                loaded_models['alignment'] = {'model': model_a, 'metadata': metadata}

            result_aligned = align(
                result["segments"],
                model_a,
                metadata,
                audio,
                device,
                interpolate_method="linear",
                print_progress=False,  # Disable printing, use callback
                combined_progress=False,  # We handle phase combination
                progress_callback=lambda p: phase_progress_callback('alignment', p),
                status_callback=status_callback
            )

            final_result["aligned"] = result_aligned

        current_phase_start += phase_weights['alignment']

        # Phase 4: Diarization (manual progress since pyannote doesn't expose it)
        if enable_diarization:
            if status_callback:
                status_callback("Identifying speakers...")

            phase_progress_callback('diarization', 10)

            # Use cached diarization model or load new one
            if cached_models and 'diarization' in cached_models:
                if status_callback:
                    status_callback("Using cached diarization model...")
                diarize_model = cached_models['diarization']
            else:
                diarize_model = DiarizationPipeline(
                    use_auth_token=kwargs.get('hf_token'),
                    device=torch.device(device)
                )
                loaded_models['diarization'] = diarize_model

            phase_progress_callback('diarization', 40)

            diarize_segments = diarize_model(audio_file)
            final_result["diarization"] = diarize_segments

            phase_progress_callback('diarization', 80)

            # Assign speakers
            result_segments = final_result.get("aligned", final_result["transcription"])
            final_result["segments_with_speakers"] = assign_word_speakers(
                diarize_segments, result_segments
            )

            phase_progress_callback('diarization', 100)

        if status_callback:
            status_callback("Transcription completed successfully")
        if progress_callback:
            progress_callback(100)

        # Return loaded models if requested (for caching)
        if return_models:
            final_result['_loaded_models'] = loaded_models

        return final_result

    except Exception as e:
        if status_callback:
            status_callback(f"Error: {str(e)}")
        raise e