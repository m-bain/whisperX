import argparse
import os
import warnings
from typing import List, Optional, Tuple, Union, Iterator, TYPE_CHECKING

import numpy as np
import torch
import tqdm
from .audio import SAMPLE_RATE, N_FRAMES, HOP_LENGTH, CHUNK_LENGTH, pad_or_trim, log_mel_spectrogram, load_audio
from .alignment import load_align_model, align, get_trellis, backtrack, merge_repeats, merge_words
from .decoding import DecodingOptions, DecodingResult
from .diarize import assign_word_speakers, Segment
from .tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
from .utils import exact_div, format_timestamp, optional_int, optional_float, str2bool, interpolate_nans, write_txt, write_vtt, write_srt, write_ass, write_tsv
from .vad import Binarize
import pandas as pd

if TYPE_CHECKING:
    from .model import Whisper



def transcribe(
    model: "Whisper",
    audio: Union[str, np.ndarray, torch.Tensor],
    *,
    verbose: Optional[bool] = None,
    temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold: Optional[float] = 2.4,
    logprob_threshold: Optional[float] = -1.0,
    no_speech_threshold: Optional[float] = 0.6,
    condition_on_previous_text: bool = False, # turn off by default due to errors it causes
    mel: np.ndarray = None,
    **decode_options,
):
    """
    Transcribe an audio file using Whisper

    Parameters
    ----------
    model: Whisper
        The Whisper model instance

    audio: Union[str, np.ndarray, torch.Tensor]
        The path to the audio file to open, or the audio waveform

    verbose: bool
        Whether to display the text being decoded to the console. If True, displays all the details,
        If False, displays minimal details. If None, does not display anything

    temperature: Union[float, Tuple[float, ...]]
        Temperature for sampling. It can be a tuple of temperatures, which will be successfully used
        upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.

    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed

    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed

    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent

    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

    decode_options: dict
        Keyword arguments to construct `DecodingOptions` instances

    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """
    dtype = torch.float16 if decode_options.get("fp16", True) else torch.float32
    if model.device == torch.device("cpu"):
        if torch.cuda.is_available():
            warnings.warn("Performing inference on CPU when CUDA is available")
        if dtype == torch.float16:
            warnings.warn("FP16 is not supported on CPU; using FP32 instead")
            dtype = torch.float32

    if dtype == torch.float32:
        decode_options["fp16"] = False

    if mel is None:
        mel = log_mel_spectrogram(audio)

    if decode_options.get("language", None) is None:
        if not model.is_multilingual:
            decode_options["language"] = "en"
        else:
            if verbose:
                print("Detecting language using up to the first 30 seconds. Use `--language` to specify the language")
            segment = pad_or_trim(mel, N_FRAMES).to(model.device).to(dtype)
            _, probs = model.detect_language(segment)
            decode_options["language"] = max(probs, key=probs.get)
            if verbose is not None:
                print(f"Detected language: {LANGUAGES[decode_options['language']].title()}")

    language = decode_options["language"]
    task = decode_options.get("task", "transcribe")
    tokenizer = get_tokenizer(model.is_multilingual, language=language, task=task)

    def decode_with_fallback(segment: torch.Tensor) -> DecodingResult:
        temperatures = [temperature] if isinstance(temperature, (int, float)) else temperature
        decode_result = None

        for t in temperatures:
            kwargs = {**decode_options}
            if t > 0:
                # disable beam_size and patience when t > 0
                kwargs.pop("beam_size", None)
                kwargs.pop("patience", None)
            else:
                # disable best_of when t == 0
                kwargs.pop("best_of", None)

            options = DecodingOptions(**kwargs, temperature=t)
            decode_result = model.decode(segment, options)

            needs_fallback = False
            if compression_ratio_threshold is not None and decode_result.compression_ratio > compression_ratio_threshold:
                needs_fallback = True  # too repetitive
            if logprob_threshold is not None and decode_result.avg_logprob < logprob_threshold:
                needs_fallback = True  # average log probability is too low

            if not needs_fallback:
                break

        return decode_result

    seek = 0
    input_stride = exact_div(
        N_FRAMES, model.dims.n_audio_ctx
    )  # mel frames per output token: 2
    time_precision = (
        input_stride * HOP_LENGTH / SAMPLE_RATE
    )  # time per output token: 0.02 (seconds)
    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    initial_prompt = decode_options.pop("initial_prompt", None) or []
    if initial_prompt:
        initial_prompt = tokenizer.encode(" " + initial_prompt.strip())
        all_tokens.extend(initial_prompt)

    def add_segment(
        *, start: float, end: float, text_tokens: torch.Tensor, result: DecodingResult
    ):
        text = tokenizer.decode([token for token in text_tokens if token < tokenizer.eot])
        if len(text.strip()) == 0:  # skip empty text output
            return

        all_segments.append(
            {
                "id": len(all_segments),
                "seek": seek,
                "start": start,
                "end": end,
                "text": text,
                "tokens": text_tokens.tolist(),
                "temperature": result.temperature,
                "avg_logprob": result.avg_logprob,
                "compression_ratio": result.compression_ratio,
                "no_speech_prob": result.no_speech_prob,
            }
        )
        if verbose:
            print(f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}")

    # show the progress bar when verbose is False (otherwise the transcribed text will be printed)
    num_frames = mel.shape[-1]
    previous_seek_value = seek

    with tqdm.tqdm(total=num_frames, unit='frames', disable=verbose is not False) as pbar:
        while seek < num_frames:
            timestamp_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
            segment = pad_or_trim(mel[:, seek:], N_FRAMES).to(model.device).to(dtype)
            segment_duration = segment.shape[-1] * HOP_LENGTH / SAMPLE_RATE

            decode_options["prompt"] = all_tokens[prompt_reset_since:]
            result: DecodingResult = decode_with_fallback(segment)
            tokens = torch.tensor(result.tokens)

            if no_speech_threshold is not None:
                # no voice activity check
                should_skip = result.no_speech_prob > no_speech_threshold
                if logprob_threshold is not None and result.avg_logprob > logprob_threshold:
                    # don't skip if the logprob is high enough, despite the no_speech_prob
                    should_skip = False

                if should_skip:
                    seek += segment.shape[-1]  # fast-forward to the next segment boundary
                    continue

            timestamp_tokens: torch.Tensor = tokens.ge(tokenizer.timestamp_begin)
            consecutive = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0].add_(1)
            if len(consecutive) > 0:  # if the output contains two consecutive timestamp tokens
                last_slice = 0
                for current_slice in consecutive:
                    sliced_tokens = tokens[last_slice:current_slice]
                    start_timestamp_position = (
                        sliced_tokens[0].item() - tokenizer.timestamp_begin
                    )
                    end_timestamp_position = (
                        sliced_tokens[-1].item() - tokenizer.timestamp_begin
                    )

                    # clamp end-time to at least be 1 frame after start-time
                    end_timestamp_position = max(end_timestamp_position, start_timestamp_position + time_precision)

                    add_segment(
                        start=timestamp_offset + start_timestamp_position * time_precision,
                        end=timestamp_offset + end_timestamp_position * time_precision,
                        text_tokens=sliced_tokens[1:-1],
                        result=result,
                    )
                    last_slice = current_slice
                last_timestamp_position = (
                    tokens[last_slice - 1].item() - tokenizer.timestamp_begin
                )
                seek += last_timestamp_position * input_stride
                all_tokens.extend(tokens[: last_slice + 1].tolist())
            else:
                duration = segment_duration
                timestamps = tokens[timestamp_tokens.nonzero().flatten()]
                if len(timestamps) > 0 and timestamps[-1].item() != tokenizer.timestamp_begin:
                    # no consecutive timestamps but it has a timestamp; use the last one.
                    # single timestamp at the end means no speech after the last timestamp.
                    last_timestamp_position = timestamps[-1].item() - tokenizer.timestamp_begin
                    duration = last_timestamp_position * time_precision

                add_segment(
                    start=timestamp_offset,
                    end=timestamp_offset + duration,
                    text_tokens=tokens,
                    result=result,
                )

                seek += segment.shape[-1]
                all_tokens.extend(tokens.tolist())

            if not condition_on_previous_text or result.temperature > 0.5:
                # do not feed the prompt tokens if a high temperature was used
                prompt_reset_since = len(all_tokens)

            # update progress bar
            pbar.update(min(num_frames, seek) - previous_seek_value)
            previous_seek_value = seek

    return dict(text=tokenizer.decode(all_tokens[len(initial_prompt):]), segments=all_segments, language=language)


def merge_chunks(segments, chunk_size=CHUNK_LENGTH):
    """
    Merge VAD segments into larger segments of approximately size ~CHUNK_LENGTH.
    TODO: Make sure VAD segment isn't too long, otherwise it will cause OOM when input to alignment model
    TODO: Or sliding window alignment model over long segment.
    """
    curr_end = 0
    merged_segments = []
    seg_idxs = []
    speaker_idxs = []

    assert chunk_size > 0
    binarize = Binarize(max_duration=chunk_size)
    segments = binarize(segments)
    segments_list = []
    for speech_turn in segments.get_timeline():
        segments_list.append(Segment(speech_turn.start, speech_turn.end, "UNKNOWN"))

    assert segments_list, "segments_list is empty."
    # Make sur the starting point is the start of the segment.
    curr_start = segments_list[0].start

    for seg in segments_list:
        if seg.end - curr_start > chunk_size and curr_end-curr_start > 0:
            merged_segments.append({
                "start": curr_start,
                "end": curr_end,
                "segments": seg_idxs,
            })
            curr_start = seg.start
            seg_idxs = []
            speaker_idxs = []
        curr_end = seg.end
        seg_idxs.append((seg.start, seg.end))
        speaker_idxs.append(seg.speaker)
    # add final
    merged_segments.append({ 
                "start": curr_start,
                "end": curr_end,
                "segments": seg_idxs,
            })    
    return merged_segments


def transcribe_with_vad(
    model: "Whisper",
    audio: Union[str, np.ndarray, torch.Tensor],
    vad_pipeline,
    mel = None,
    verbose: Optional[bool] = None,
    **kwargs
):
    """
    Transcribe per VAD segment
    """

    if mel is None:
        mel = log_mel_spectrogram(audio)
    
    prev = 0
    output = {"segments": []}

    vad_segments = vad_pipeline(audio)
    # merge segments to approx 30s inputs to make whisper most appropraite
    vad_segments = merge_chunks(vad_segments)

    for sdx, seg_t in enumerate(vad_segments):
        if verbose:
            print(f"~~ Transcribing VAD chunk: ({format_timestamp(seg_t['start'])} --> {format_timestamp(seg_t['end'])}) ~~")
        seg_f_start, seg_f_end = int(seg_t["start"] * SAMPLE_RATE / HOP_LENGTH), int(seg_t["end"] * SAMPLE_RATE / HOP_LENGTH)
        local_f_start, local_f_end = seg_f_start - prev, seg_f_end - prev
        mel = mel[:, local_f_start:] # seek forward
        prev = seg_f_start
        local_mel = mel[:, :local_f_end-local_f_start]
        result = transcribe(model, audio, mel=local_mel, verbose=verbose, **kwargs)
        seg_t["text"] = result["text"]
        output["segments"].append(
            {
                "start": seg_t["start"],
                "end": seg_t["end"],
                "language": result["language"],
                "text": result["text"],
                "seg-text": [x["text"] for x in result["segments"]],
                "seg-start": [x["start"] for x in result["segments"]],
                "seg-end": [x["end"] for x in result["segments"]],
                }
            )

    output["language"] = output["segments"][0]["language"]

    return output


def transcribe_with_vad_parallel(
    model: "Whisper",
    audio: Union[str, np.ndarray, torch.Tensor],
    vad_pipeline,
    mel = None,
    verbose: Optional[bool] = None,
    batch_size = -1,
    **kwargs
):
    """
    Transcribe per VAD segment
    """

    if mel is None:
        mel = log_mel_spectrogram(audio)
    
    vad_segments = vad_pipeline(audio)
    # merge segments to approx 30s inputs to make whisper most appropraite
    vad_segments = merge_chunks(vad_segments)

    ################################
    ### START of parallelization ###
    ################################

    # pad mel to a same length
    start_seconds = [i['start'] for i in vad_segments]
    end_seconds = [i['end'] for i in vad_segments]
    duration_list = np.array(end_seconds) - np.array(start_seconds)
    max_length = round(30 / (HOP_LENGTH / SAMPLE_RATE))
    offset_list = np.array(start_seconds)
    chunks = []

    for start_ts, end_ts in zip(start_seconds, end_seconds):
        start_ts = round(start_ts / (HOP_LENGTH / SAMPLE_RATE))
        end_ts = round(end_ts / (HOP_LENGTH / SAMPLE_RATE))
        chunk = mel[:, start_ts:end_ts]
        chunk = torch.nn.functional.pad(chunk, (0, max_length-chunk.shape[-1]))
        chunks.append(chunk)
    
    mel_chunk = torch.stack(chunks, dim=0).to(model.device)
    # using 'decode_options1': only support single temperature decoding (no fallbacks)
    # result_list2 = model.decode(mel_chunk, decode_options1)

    # prepare DecodingOptions
    temperatures = kwargs.pop("temperature", None)
    compression_ratio_threshold = kwargs.pop("compression_ratio_threshold", None)
    logprob_threshold = kwargs.pop("logprob_threshold", None)
    no_speech_threshold = kwargs.pop("no_speech_threshold", None)
    condition_on_previous_text = kwargs.pop("condition_on_previous_text", None)
    initial_prompt = kwargs.pop("initial_prompt", None)
    
    t = 0  # TODO: does not upport temperature sweeping
    if t > 0:
        # disable beam_size and patience when t > 0
        kwargs.pop("beam_size", None)
        kwargs.pop("patience", None)
    else:
        # disable best_of when t == 0
        kwargs.pop("best_of", None)

    options = DecodingOptions(**kwargs, temperature=t)
    mel_chunk_batches = torch.split(mel_chunk, split_size_or_sections=batch_size)
    decode_result = []
    for mel_chunk_batch in mel_chunk_batches:
        decode_result.extend(model.decode(mel_chunk_batch, options))
    
    ##############################
    ### END of parallelization ###
    ##############################

    # post processing: get segments rfom batch-decoded results
    input_stride = exact_div(
        N_FRAMES, model.dims.n_audio_ctx
    )  # mel frames per output token: 2
    language = kwargs["language"]
    task = kwargs["task"]
    tokenizer = get_tokenizer(model.is_multilingual, language=language, task=task)

    output = post_process_results(
        vad_segments,
        decode_result, 
        duration_list, 
        offset_list,
        input_stride,
        language,
        tokenizer,
        no_speech_threshold=no_speech_threshold,
        logprob_threshold=logprob_threshold,
        verbose=verbose)
    return output


def post_process_results(
    vad_segments,
    result_list, 
    duration_list, 
    offset_list,
    input_stride,
    language,
    tokenizer,
    no_speech_threshold = None,
    logprob_threshold = None,
    verbose: Optional[bool] = None,
    ):

    seek = 0
    time_precision = (
        input_stride * HOP_LENGTH / SAMPLE_RATE
    )  # time per output token: 0.02 (seconds)
    all_tokens = []
    all_segments = []
    output = {"segments": []}

    def add_segment(
        *, start: float, end: float, text_tokens: torch.Tensor, result: DecodingResult
    ):
        text = tokenizer.decode([token for token in text_tokens if token < tokenizer.eot])
        if len(text.strip()) == 0:  # skip empty text output
            return

        all_segments.append(
            {
                "id": len(all_segments),
                "seek": seek,
                "start": start,
                "end": end,
                "text": text,
                "tokens": text_tokens.tolist(),
                "temperature": result.temperature,
                "avg_logprob": result.avg_logprob,
                "compression_ratio": result.compression_ratio,
                "no_speech_prob": result.no_speech_prob,
            }
        )
        if verbose:
            print(f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}")

    # process the output
    for seg_t, result, segment_duration, timestamp_offset in zip(vad_segments, result_list, duration_list, offset_list):
        all_tokens = []
        all_segments = []

        # segment_duration = segment.shape[-1] * HOP_LENGTH / SAMPLE_RATE
        segment_shape = int(segment_duration / (HOP_LENGTH / SAMPLE_RATE))
        tokens = torch.tensor(result.tokens)

        if no_speech_threshold is not None:
            # no voice activity check
            should_skip = result.no_speech_prob > no_speech_threshold
            if logprob_threshold is not None and result.avg_logprob > logprob_threshold:
                # don't skip if the logprob is high enough, despite the no_speech_prob
                should_skip = False

            if should_skip:
                seek += segment_shape  # fast-forward to the next segment boundary
                continue
        
        timestamp_tokens: torch.Tensor = tokens.ge(tokenizer.timestamp_begin)
        consecutive = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0].add_(1)
        
        if len(consecutive) > 0:  # if the output contains two consecutive timestamp tokens
            last_slice = 0
            for current_slice in consecutive:
                sliced_tokens = tokens[last_slice:current_slice]
                start_timestamp_position = (
                    sliced_tokens[0].item() - tokenizer.timestamp_begin
                )
                end_timestamp_position = (
                    sliced_tokens[-1].item() - tokenizer.timestamp_begin
                )
                add_segment(
                    start=timestamp_offset + start_timestamp_position * time_precision,
                    end=timestamp_offset + end_timestamp_position * time_precision,
                    text_tokens=sliced_tokens[1:-1],
                    result=result,
                )
                last_slice = current_slice
            last_timestamp_position = (
                tokens[last_slice - 1].item() - tokenizer.timestamp_begin
            )
            seek += last_timestamp_position * input_stride
            all_tokens.extend(tokens[: last_slice + 1].tolist())
        else:
            duration = segment_duration
            timestamps = tokens[timestamp_tokens.nonzero().flatten()]
            if len(timestamps) > 0 and timestamps[-1].item() != tokenizer.timestamp_begin:
                # no consecutive timestamps but it has a timestamp; use the last one.
                # single timestamp at the end means no speech after the last timestamp.
                last_timestamp_position = timestamps[-1].item() - tokenizer.timestamp_begin
                duration = last_timestamp_position * time_precision

            add_segment(
                start=timestamp_offset,
                end=timestamp_offset + duration,
                text_tokens=tokens,
                result=result,
            )

            seek += segment_shape
            all_tokens.extend(tokens.tolist())

        result = dict(text=tokenizer.decode(all_tokens), segments=all_segments, language=language)
        output["segments"].append(
            {
                "start": seg_t["start"],
                "end": seg_t["end"],
                "language": result["language"],
                "text": result["text"],
                "seg-text": [x["text"] for x in result["segments"]],
                "seg-start": [x["start"] for x in result["segments"]],
                "seg-end": [x["end"] for x in result["segments"]],
                }
            )

    output["language"] = output["segments"][0]["language"]

    return output


def cli():
    from . import available_models

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio", nargs="+", type=str, help="audio file(s) to transcribe")
    parser.add_argument("--model", default="small", choices=available_models(), help="name of the Whisper model to use")
    parser.add_argument("--model_dir", type=str, default=None, help="the path to save model files; uses ~/.cache/whisper by default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device to use for PyTorch inference")
    # alignment params
    parser.add_argument("--align_model", default=None, help="Name of phoneme-level ASR model to do alignment")
    parser.add_argument("--align_extend", default=2, type=float, help="Seconds before and after to extend the whisper segments for alignment")
    parser.add_argument("--align_from_prev", default=True, type=bool, help="Whether to clip the alignment start time of current segment to the end time of the last aligned word of the previous segment")
    parser.add_argument("--interpolate_method", default="nearest", choices=["nearest", "linear", "ignore"], help="For word .srt, method to assign timestamps to non-aligned words, or merge them into neighbouring.")
    # vad params
    parser.add_argument("--vad_filter", action="store_true", help="Whether to first perform VAD filtering to target only transcribe within VAD. Produces more accurate alignment + timestamp, requires more GPU memory & compute.")
    parser.add_argument("--parallel_bs", default=-1, type=int, help="Enable parallel transcribing if > 1")
    # diarization params
    parser.add_argument("--diarize", action="store_true", help="Apply diarization to assign speaker labels to each segment/word")
    parser.add_argument("--min_speakers", default=None, type=int)
    parser.add_argument("--max_speakers", default=None, type=int)
    # output save params
    parser.add_argument("--output_dir", "-o", type=str, default=".", help="directory to save the outputs")
    parser.add_argument("--output_type", default="all", choices=["all", "srt", "srt-word", "vtt", "txt", "tsv", "ass", "ass-char", "pickle", "vad"], help="File type for desired output save")

    parser.add_argument("--verbose", type=str2bool, default=True, help="whether to print out the progress and debug messages")

    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default=None, choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]), help="language spoken in the audio, specify None to perform language detection")

    parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
    parser.add_argument("--best_of", type=optional_int, default=5, help="number of candidates when sampling with non-zero temperature")
    parser.add_argument("--beam_size", type=optional_int, default=5, help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=None, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=None, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

    parser.add_argument("--suppress_tokens", type=str, default="-1", help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    parser.add_argument("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
    parser.add_argument("--condition_on_previous_text", type=str2bool, default=False, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")
    parser.add_argument("--fp16", type=str2bool, default=True, help="whether to perform inference in fp16; True by default")

    parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")
    parser.add_argument("--threads", type=optional_int, default=0, help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face Access Token to access PyAnnote gated models")
    
    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    model_dir: str = args.pop("model_dir")
    output_dir: str = args.pop("output_dir")
    output_type: str = args.pop("output_type")
    device: str = args.pop("device")

    align_model: str = args.pop("align_model")
    align_extend: float = args.pop("align_extend")
    align_from_prev: bool = args.pop("align_from_prev")
    interpolate_method: bool = args.pop("interpolate_method")
    
    hf_token: str = args.pop("hf_token")
    vad_filter: bool = args.pop("vad_filter")
    parallel_bs: int = args.pop("parallel_bs")

    diarize: bool = args.pop("diarize")
    min_speakers: int = args.pop("min_speakers")
    max_speakers: int = args.pop("max_speakers")

    vad_pipeline = None
    if vad_filter:
        if hf_token is None:
            print("Warning, no huggingface token used, needs to be saved in environment variable, otherwise will throw error loading VAD model...")
        from pyannote.audio import Inference, Model
        vad_pipeline = Inference(
            Model.from_pretrained("pyannote/segmentation",
                                  use_auth_token=hf_token),
            pre_aggregation_hook=lambda segmentation: segmentation,
            use_auth_token=hf_token,
            device=torch.device(device),
        )

    diarize_pipeline = None
    if diarize:
        if hf_token is None:
            print("Warning, no --hf_token used, needs to be saved in environment variable, otherwise will throw error loading diarization model...")
        from pyannote.audio import Pipeline
        diarize_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token=hf_token)

    os.makedirs(output_dir, exist_ok=True)

    if model_name.endswith(".en") and args["language"] not in {"en", "English"}:
        if args["language"] is not None:
            warnings.warn(f'{model_name} is an English-only model but receipted "{args["language"]}"; using English instead.')
        args["language"] = "en"

    temperature = args.pop("temperature")
    temperature_increment_on_fallback = args.pop("temperature_increment_on_fallback")
    if temperature_increment_on_fallback is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback))
    else:
        temperature = [temperature]

    threads = args.pop("threads")
    if threads > 0:
        torch.set_num_threads(threads)

    from . import load_model
    model = load_model(model_name, device=device, download_root=model_dir)

    align_language = args["language"] if args["language"] is not None else "en" # default to loading english if not specified
    align_model, align_metadata = load_align_model(align_language, device, model_name=align_model)

    for audio_path in args.pop("audio"):
        if vad_filter:
            if parallel_bs > 1:
                print("Performing VAD and parallel transcribing ...")
                result = transcribe_with_vad_parallel(model, audio_path, vad_pipeline, temperature=temperature, batch_size=parallel_bs, **args)
            else:
                print("Performing VAD...")
                result = transcribe_with_vad(model, audio_path, vad_pipeline, temperature=temperature, **args)
        else:
            print("Performing transcription...")
            result = transcribe(model, audio_path, temperature=temperature, **args)

        if result["language"] != align_metadata["language"]:
            # load new language
            print(f"New language found ({result['language']})! Previous was ({align_metadata['language']}), loading new alignment model for new language...")
            align_model, align_metadata = load_align_model(result["language"], device)


        print("Performing alignment...")
        result_aligned = align(result["segments"], align_model, align_metadata, audio_path, device,
                                extend_duration=align_extend, start_from_previous=align_from_prev, interpolate_method=interpolate_method)
        audio_basename = os.path.basename(audio_path)

        if diarize:
            print("Performing diarization...")
            diarize_segments = diarize_pipeline(audio_path, min_speakers=min_speakers, max_speakers=max_speakers)
            diarize_df = pd.DataFrame(diarize_segments.itertracks(yield_label=True))
            diarize_df['start'] = diarize_df[0].apply(lambda x: x.start)
            diarize_df['end'] = diarize_df[0].apply(lambda x: x.end)
            # assumes each utterance is single speaker (needs fix)
            result_segments, word_segments = assign_word_speakers(diarize_df, result_aligned["segments"], fill_nearest=True)
            result_aligned["segments"] = result_segments
            result_aligned["word_segments"] = word_segments

        # save TXT
        if output_type in ["txt", "all"]:
            with open(os.path.join(output_dir, audio_basename + ".txt"), "w", encoding="utf-8") as txt:
                write_txt(result_aligned["segments"], file=txt)

        # save VTT
        if output_type in ["vtt", "all"]:
            with open(os.path.join(output_dir, audio_basename + ".vtt"), "w", encoding="utf-8") as vtt:
                write_vtt(result_aligned["segments"], file=vtt)

        # save SRT
        if output_type in ["srt", "all"]:
            with open(os.path.join(output_dir, audio_basename + ".srt"), "w", encoding="utf-8") as srt:
                write_srt(result_aligned["segments"], file=srt)

        # save TSV
        if output_type in ["tsv", "all"]:
            with open(os.path.join(output_dir, audio_basename + ".tsv"), "w", encoding="utf-8") as srt:
                write_tsv(result_aligned["segments"], file=srt)

        # save SRT word-level
        if output_type in ["srt-word", "all"]:
            # save per-word SRT
            with open(os.path.join(output_dir, audio_basename + ".word.srt"), "w", encoding="utf-8") as srt:
                write_srt(result_aligned["word_segments"], file=srt)

        # save ASS
        if output_type in ["ass", "all"]:
            with open(os.path.join(output_dir, audio_basename + ".ass"), "w", encoding="utf-8") as ass:
                write_ass(result_aligned["segments"], file=ass)
        
        # # save ASS character-level
        if output_type in ["ass-char"]:
            with open(os.path.join(output_dir, audio_basename + ".char.ass"), "w", encoding="utf-8") as ass:
                write_ass(result_aligned["segments"], file=ass, resolution="char")

        # save word tsv
        if output_type in ["pickle"]:
            exp_fp = os.path.join(output_dir, audio_basename + ".pkl")
            pd.DataFrame(result_aligned["segments"]).to_pickle(exp_fp)

        # save word tsv
        if output_type in ["vad"]:
            exp_fp = os.path.join(output_dir, audio_basename + ".sad")
            wrd_segs = pd.concat([x["word-segments"] for x in result_aligned["segments"]])[['start','end']]
            wrd_segs.to_csv(exp_fp, sep='\t', header=None, index=False)
if __name__ == "__main__":
    cli()
