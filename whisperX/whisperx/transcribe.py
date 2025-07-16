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
from whisperx.types import AlignedTranscriptionResult, TranscriptionResult
from whisperx.utils import LANGUAGES, TO_LANGUAGE_CODE, get_writer


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
    tmp_results = []
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
        print(">>Performing transcription...")
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
                    print(
                        f"New language found ({result['language']})! Previous was ({align_metadata['language']}), loading new alignment model for new language..."
                    )
                    align_model, align_metadata = load_align_model(
                        result["language"], device
                    )
                print(">>Performing alignment...")
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
            print(
                "Warning, no --hf_token used, needs to be saved in environment variable, otherwise will throw error loading diarization model..."
            )
        tmp_results = results
        print(">>Performing diarization...")
        print(">>Using model:", diarize_model_name)
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
