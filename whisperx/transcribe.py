import argparse
import os
import warnings
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
import torch

from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
from whisper.utils import (
    optional_float,
    optional_int,
    str2bool,
)

from .utils import get_writer

from .asr import transcribe, transcribe_with_vad
from .alignment import load_align_model, align
from .diarize import DiarizationPipeline
from .vad import load_vad_model

def cli():
    from whisper import available_models

    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio", nargs="+", type=str, help="audio file(s) to transcribe")
    parser.add_argument("--model", default="small", choices=available_models(), help="name of the Whisper model to use")
    parser.add_argument("--model_dir", type=str, default=None, help="the path to save model files; uses ~/.cache/whisper by default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device to use for PyTorch inference")
    parser.add_argument("--output_dir", "-o", type=str, default=".", help="directory to save the outputs")
    parser.add_argument("--output_format", "-f", type=str, default="all", choices=["all", "srt", "srt-word", "vtt", "txt", "tsv", "ass", "ass-char", "pickle", "vad"], help="format of the output file; if not specified, all available formats will be produced")
    parser.add_argument("--verbose", type=str2bool, default=True, help="whether to print out the progress and debug messages")

    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default=None, choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]), help="language spoken in the audio, specify None to perform language detection")

    # alignment params
    parser.add_argument("--align_model", default=None, help="Name of phoneme-level ASR model to do alignment")
    parser.add_argument("--align_extend", default=2, type=float, help="Seconds before and after to extend the whisper segments for alignment (if not using VAD).")
    parser.add_argument("--align_from_prev", default=True, type=bool, help="Whether to clip the alignment start time of current segment to the end time of the last aligned word of the previous segment (if not using VAD)")
    parser.add_argument("--interpolate_method", default="nearest", choices=["nearest", "linear", "ignore"], help="For word .srt, method to assign timestamps to non-aligned words, or merge them into neighbouring.")
    parser.add_argument("--no_align", action='store_true', help="Do not perform phoneme alignment")

    # vad params
    parser.add_argument("--vad_filter", action="store_true", help="Whether to pre-segment audio with VAD, highly recommended! Produces more accurate alignment + timestamp see WhisperX paper https://arxiv.org/abs/2303.00747")
    parser.add_argument("--vad_onset", type=float, default=0.767, help="Onset threshold for VAD (see pyannote.audio)")
    parser.add_argument("--vad_offset", type=float, default=0.363, help="Offset threshold for VAD (see pyannote.audio).")

    # diarization params
    parser.add_argument("--diarize", action="store_true", help="Apply diarization to assign speaker labels to each segment/word")
    parser.add_argument("--min_speakers", default=None, type=int)
    parser.add_argument("--max_speakers", default=None, type=int)

    parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
    parser.add_argument("--best_of", type=optional_int, default=5, help="number of candidates when sampling with non-zero temperature")
    parser.add_argument("--beam_size", type=optional_int, default=5, help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=None, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=None, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

    parser.add_argument("--suppress_tokens", type=str, default="-1", help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    parser.add_argument("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
    parser.add_argument("--condition_on_previous_text", type=str2bool, default=True, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")
    parser.add_argument("--fp16", type=str2bool, default=True, help="whether to perform inference in fp16; True by default")

    parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")
    parser.add_argument("--word_timestamps", type=str2bool, default=False, help="(experimental) extract word-level timestamps and refine the results based on them")
    parser.add_argument("--prepend_punctuations", type=str, default="\"\'“¿([{-", help="if word_timestamps is True, merge these punctuation symbols with the next word")
    parser.add_argument("--append_punctuations", type=str, default="\"\'.。,，!！?？:：”)]}、", help="if word_timestamps is True, merge these punctuation symbols with the previous word")
    parser.add_argument("--threads", type=optional_int, default=0, help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")

    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face Access Token to access PyAnnote gated models")
    parser.add_argument("--model_flush", action="store_true", help="Flush memory of each stage after use, more GPU memory efficient, but slower when there are multiple audio files")
    # fmt: on

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    model_dir: str = args.pop("model_dir")
    output_dir: str = args.pop("output_dir")
    output_format: str = args.pop("output_format")
    device: str = args.pop("device")
    model_flush: bool = args.pop("model_flush")
    os.makedirs(output_dir, exist_ok=True)

    align_model: str = args.pop("align_model")
    align_extend: float = args.pop("align_extend")
    align_from_prev: bool = args.pop("align_from_prev")
    interpolate_method: str = args.pop("interpolate_method")
    no_align: bool = args.pop("no_align")

    hf_token: str = args.pop("hf_token")
    vad_filter: bool = args.pop("vad_filter")
    vad_onset: float = args.pop("vad_onset")
    vad_offset: float = args.pop("vad_offset")

    diarize: bool = args.pop("diarize")
    min_speakers: int = args.pop("min_speakers")
    max_speakers: int = args.pop("max_speakers")

    if vad_filter:
        if hf_token is None:
            print("Warning, no --hf_token used, needs to be saved in environment variable, otherwise will throw error loading diarization model...")
        from pyannote.audio import Pipeline
        from pyannote.audio import Model, Pipeline
        vad_model = load_vad_model(torch.device(device), vad_onset, vad_offset, use_auth_token=hf_token)
    else:
        vad_model = None

    if diarize:
        if hf_token is None:
            print("Warning, no --hf_token used, needs to be saved in environment variable, otherwise will throw error loading diarization model...")
        diarize_model = DiarizationPipeline(use_auth_token=hf_token)
    else:
        diarize_model = None

    if no_align:
        align_model, align_metadata = None, None
    else:
        align_language = args["language"] if args["language"] is not None else "en" # default to loading english if not specified
        align_model, align_metadata = load_align_model(align_language, device, model_name=align_model)     


    if model_name.endswith(".en") and args["language"] not in {"en", "English"}:
        if args["language"] is not None:
            warnings.warn(
                f"{model_name} is an English-only model but receipted '{args['language']}'; using English instead."
            )
        args["language"] = "en"

    temperature = args.pop("temperature")
    if (increment := args.pop("temperature_increment_on_fallback")) is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = [temperature]

    if (threads := args.pop("threads")) > 0:
        torch.set_num_threads(threads)

    from whisper import load_model

    model = load_model(model_name, device=device, download_root=model_dir)

    writer = get_writer(output_format, output_dir)
    for audio_path in args.pop("audio"):

        if vad_model is not None:
            print("Performing VAD...")
            result = transcribe_with_vad(model, audio_path, vad_model, temperature=temperature, **args)
        else:
            print("Performing transcription...")
            result = transcribe(model, audio_path, temperature=temperature, **args)

        if align_model is not None:
            if result["language"] != align_metadata["language"]:
                # load new language
                print(f"New language found ({result['language']})! Previous was ({align_metadata['language']}), loading new alignment model for new language...")
                align_model, align_metadata = load_align_model(result["language"], device)

            result = align(result["segments"], align_model, align_metadata, audio_path, device,
                extend_duration=align_extend, start_from_previous=align_from_prev, interpolate_method=interpolate_method)

        # if diarize_model is not None:
        #     diarize_segments = diarize_model(audio_path, min_speakers=min_speakers, max_speakers=max_speakers)
        #     results_segments, word_segments = assign_word_speakers(diarize_segments, )

        writer(result, audio_path)


if __name__ == "__main__":
    cli()