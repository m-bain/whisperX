import argparse
import os
import warnings
from typing import List, Optional, Tuple, Union, Iterator, TYPE_CHECKING

import numpy as np
import torch
import torchaudio
from transformers import AutoProcessor, Wav2Vec2ForCTC
import tqdm
from .audio import SAMPLE_RATE, N_FRAMES, HOP_LENGTH, CHUNK_LENGTH, pad_or_trim, log_mel_spectrogram, load_audio
from .alignment import get_trellis, backtrack, merge_repeats, merge_words
from .decoding import DecodingOptions, DecodingResult
from .tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
from .utils import exact_div, format_timestamp, optional_int, optional_float, str2bool, interpolate_nans, write_txt, write_vtt, write_srt, write_ass, write_tsv
import pandas as pd

if TYPE_CHECKING:
    from .model import Whisper

LANGUAGES_WITHOUT_SPACES = ["ja", "zh"]

DEFAULT_ALIGN_MODELS_TORCH = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "fr": "VOXPOPULI_ASR_BASE_10K_FR",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "es": "VOXPOPULI_ASR_BASE_10K_ES",
    "it": "VOXPOPULI_ASR_BASE_10K_IT",
}

DEFAULT_ALIGN_MODELS_HF = {
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "uk": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
}


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


def align(
    transcript: Iterator[dict],
    model: torch.nn.Module,
    align_model_metadata: dict,
    audio: Union[str, np.ndarray, torch.Tensor],
    device: str,
    extend_duration: float = 0.0,
    start_from_previous: bool = True,
    interpolate_method: str = "nearest",
):
    """
    Force align phoneme recognition predictions to known transcription

    Parameters
    ----------
    transcript: Iterator[dict]
        The Whisper model instance
    
    model: torch.nn.Module
        Alignment model (wav2vec2)

    audio: Union[str, np.ndarray, torch.Tensor]
        The path to the audio file to open, or the audio waveform

    device: str
        cuda device

    extend_duration: float
        Amount to pad input segments by. If not using vad--filter then recommended to use 2 seconds

        If the gzip compression ratio is above this value, treat as failed

    interpolate_method: str ["nearest", "linear", "ignore"]
        Method to assign timestamps to non-aligned words. Words are not able to be aligned when none of the characters occur in the align model dictionary.
        "nearest" copies timestamp of nearest word within the segment. "linear" is linear interpolation. "drop" removes that word from output.

    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)

    MAX_DURATION = audio.shape[1] / SAMPLE_RATE

    model_dictionary = align_model_metadata["dictionary"]
    model_lang = align_model_metadata["language"]
    model_type = align_model_metadata["type"]

    aligned_segments = []

    prev_t2 = 0
    sdx = 0
    for segment in transcript:
        while True:
            segment_align_success = False

            # strip spaces at beginning / end, but keep track of the amount.
            num_leading = len(segment["text"]) - len(segment["text"].lstrip())
            num_trailing = len(segment["text"]) - len(segment["text"].rstrip())
            transcription = segment["text"]

            # TODO: convert number tokenizer / symbols to phonetic words for alignment.
            # e.g. "$300" -> "three hundred dollars"
            # currently "$300" is ignored since no characters present in the phonetic dictionary

            # split into words
            if model_lang not in LANGUAGES_WITHOUT_SPACES:
                per_word = transcription.split(" ")
            else:
                per_word = transcription

            # first check that characters in transcription can be aligned (they are contained in align model"s dictionary)
            clean_char, clean_cdx = [], []
            for cdx, char in enumerate(transcription):
                char_ = char.lower()
                # wav2vec2 models use "|" character to represent spaces
                if model_lang not in LANGUAGES_WITHOUT_SPACES:
                    char_ = char_.replace(" ", "|")
                
                # ignore whitespace at beginning and end of transcript
                if cdx < num_leading:
                    pass
                elif cdx > len(transcription) - num_trailing - 1:
                    pass
                elif char_ in model_dictionary.keys():
                    clean_char.append(char_)
                    clean_cdx.append(cdx)

            clean_wdx = []
            for wdx, wrd in enumerate(per_word):
                if any([c in model_dictionary.keys() for c in wrd]):
                    clean_wdx.append(wdx)

            # if no characters are in the dictionary, then we skip this segment...
            if len(clean_char) == 0:
                print("Failed to align segment: no characters in this segment found in model dictionary, resorting to original...")
                break          
           
            transcription_cleaned = "".join(clean_char)
            tokens = [model_dictionary[c] for c in transcription_cleaned]

            # pad according original timestamps
            t1 = max(segment["start"] - extend_duration, 0)
            t2 = min(segment["end"] + extend_duration, MAX_DURATION)

            # use prev_t2 as current t1 if it"s later
            if start_from_previous and t1 < prev_t2:
                t1 = prev_t2

            # check if timestamp range is still valid
            if t1 >= MAX_DURATION:
                print("Failed to align segment: original start time longer than audio duration, skipping...")
                break
            if t2 - t1 < 0.02:
                print("Failed to align segment: duration smaller than 0.02s time precision")
                break

            f1 = int(t1 * SAMPLE_RATE)
            f2 = int(t2 * SAMPLE_RATE)

            waveform_segment = audio[:, f1:f2]

            with torch.inference_mode():
                if model_type == "torchaudio":
                    emissions, _ = model(waveform_segment.to(device))
                elif model_type == "huggingface":
                    emissions = model(waveform_segment.to(device)).logits
                else:
                    raise NotImplementedError(f"Align model of type {model_type} not supported.")
                emissions = torch.log_softmax(emissions, dim=-1)

            emission = emissions[0].cpu().detach()

            trellis = get_trellis(emission, tokens)
            path = backtrack(trellis, emission, tokens)
            if path is None:
                print("Failed to align segment: backtrack failed, resorting to original...")
                break
            char_segments = merge_repeats(path, transcription_cleaned)
            # word_segments = merge_words(char_segments)
            

            # sub-segments
            if "seg-text" not in segment:
                segment["seg-text"] = [transcription]
                
            v = 0
            seg_lens = [0] + [len(x) for x in segment["seg-text"]]
            seg_lens_cumsum = [v := v + n for n in seg_lens]
            sub_seg_idx = 0

            char_level = {
                "start": [],
                "end": [],
                "score": [],
                "word-index": [],
            }

            word_level = {
                "start": [],
                "end": [],
                "score": [],
                "segment-text-start": [],
                "segment-text-end": []
            }

            wdx = 0
            seg_start_actual, seg_end_actual = None, None
            duration = t2 - t1
            ratio = duration * waveform_segment.size(0) / (trellis.size(0) - 1)
            cdx_prev = 0
            for cdx, char in enumerate(transcription + " "):
                is_last = False
                if cdx == len(transcription):
                    break
                elif cdx+1 == len(transcription):
                    is_last = True

                
                start, end, score = None, None, None
                if cdx in clean_cdx:
                    char_seg = char_segments[clean_cdx.index(cdx)]
                    start = char_seg.start * ratio + t1
                    end = char_seg.end * ratio + t1
                    score = char_seg.score        

                char_level["start"].append(start)
                char_level["end"].append(end)
                char_level["score"].append(score)
                char_level["word-index"].append(wdx)

                # word-level info
                if model_lang in LANGUAGES_WITHOUT_SPACES:
                    # character == word
                    wdx += 1
                elif is_last or transcription[cdx+1] == " " or cdx == seg_lens_cumsum[sub_seg_idx+1] - 1:
                    wdx += 1
                    word_level["start"].append(None)
                    word_level["end"].append(None)
                    word_level["score"].append(None)
                    word_level["segment-text-start"].append(cdx_prev-seg_lens_cumsum[sub_seg_idx])
                    word_level["segment-text-end"].append(cdx+1-seg_lens_cumsum[sub_seg_idx])
                    cdx_prev = cdx+2

                if is_last or cdx == seg_lens_cumsum[sub_seg_idx+1] - 1:
                    if model_lang not in LANGUAGES_WITHOUT_SPACES:
                        char_level = pd.DataFrame(char_level)
                        word_level = pd.DataFrame(word_level)
                        
                        not_space = pd.Series(list(segment["seg-text"][sub_seg_idx])) != " "
                        word_level["start"] = char_level[not_space].groupby("word-index")["start"].min() # take min of all chars in a word ignoring space
                        word_level["end"] = char_level[not_space].groupby("word-index")["end"].max() # take max of all chars in a word

                        # fill missing
                        if interpolate_method != "ignore":
                            word_level["start"] = interpolate_nans(word_level["start"], method=interpolate_method) 
                            word_level["end"] = interpolate_nans(word_level["end"], method=interpolate_method)
                        word_level["start"] = word_level["start"].values.tolist()
                        word_level["end"] = word_level["end"].values.tolist()
                        word_level["score"] = char_level.groupby("word-index")["score"].mean() # take mean of all scores

                        char_level = char_level.replace({np.nan:None}).to_dict("list")
                        word_level = pd.DataFrame(word_level).replace({np.nan:None}).to_dict("list")
                    else:
                        word_level = None

                    aligned_segments.append(
                        {
                            "text": segment["seg-text"][sub_seg_idx],
                            "start": seg_start_actual,
                            "end": seg_end_actual,
                            "char-segments": char_level,
                            "word-segments": word_level
                        }
                    )
                    if "language" in segment:
                        aligned_segments[-1]["language"] = segment["language"]

                    print(f"[{format_timestamp(aligned_segments[-1]['start'])} --> {format_timestamp(aligned_segments[-1]['end'])}] {aligned_segments[-1]['text']}")


                    char_level = {
                        "start": [],
                        "end": [],
                        "score": [],
                        "word-index": [],
                    }
                    word_level = {
                        "start": [],
                        "end": [],
                        "score": [],
                        "segment-text-start": [],
                        "segment-text-end": []
                    }
                    wdx = 0
                    cdx_prev = cdx + 2
                    sub_seg_idx += 1
                    seg_start_actual, seg_end_actual = None, None


                # take min-max for actual segment-level timestamp
                if seg_start_actual is None and start is not None:
                    seg_start_actual = start
                if end is not None:
                    seg_end_actual = end


            prev_t2 = segment["end"]

            segment_align_success = True
            # end while True loop
            break

        # reset prev_t2 due to drifting issues
        if not segment_align_success:
            prev_t2 = 0

        # shift segment index by amount of sub-segments
        if "seg-text" in segment:
            sdx += len(segment["seg-text"])
        else:
            sdx += 1

    # create word level segments for .srt
    word_seg = []
    for seg in aligned_segments:
        if model_lang in LANGUAGES_WITHOUT_SPACES:
            # character based
            seg["word-segments"] = seg["char-segments"]
            seg["word-segments"]["segment-text-start"] = range(len(seg['word-segments']['start']))
            seg["word-segments"]["segment-text-end"] = range(1, len(seg['word-segments']['start'])+1)

        wseg = pd.DataFrame(seg["word-segments"]).replace({np.nan:None})
        for wdx, wrow in wseg.iterrows():
            if wrow["start"] is not None:
                word_seg.append(
                    {
                        "start": wrow["start"],
                        "end": wrow["end"],
                        "text": seg["text"][int(wrow["segment-text-start"]):int(wrow["segment-text-end"])]
                    }
                )

    return {"segments": aligned_segments, "word_segments": word_seg}

def load_align_model(language_code, device, model_name=None):
    if model_name is None:
        # use default model
        if language_code in DEFAULT_ALIGN_MODELS_TORCH:
            model_name = DEFAULT_ALIGN_MODELS_TORCH[language_code]
        elif language_code in DEFAULT_ALIGN_MODELS_HF:
            model_name = DEFAULT_ALIGN_MODELS_HF[language_code]
        else:
            print(f"There is no default alignment model set for this language ({language_code}).\
                Please find a wav2vec2.0 model finetuned on this language in https://huggingface.co/models, then pass the model name in --align_model [MODEL_NAME]")
            raise ValueError(f"No default align-model for language: {language_code}")

    if model_name in torchaudio.pipelines.__all__:
        pipeline_type = "torchaudio"
        bundle = torchaudio.pipelines.__dict__[model_name]
        align_model = bundle.get_model().to(device)
        labels = bundle.get_labels()
        align_dictionary = {c.lower(): i for i, c in enumerate(labels)}
    else:
        try:
            processor = AutoProcessor.from_pretrained(model_name)
            align_model = Wav2Vec2ForCTC.from_pretrained(model_name)
        except Exception as e:
            print(e)
            print(f"Error loading model from huggingface, check https://huggingface.co/models for finetuned wav2vec2.0 models")
            raise ValueError(f'The chosen align_model "{model_name}" could not be found in huggingface (https://huggingface.co/models) or torchaudio (https://pytorch.org/audio/stable/pipelines.html#id14)')
        pipeline_type = "huggingface"
        align_model = align_model.to(device)
        labels = processor.tokenizer.get_vocab()
        align_dictionary = {char.lower(): code for char,code in processor.tokenizer.get_vocab().items()}

    align_metadata = {"language": language_code, "dictionary": align_dictionary, "type": pipeline_type}

    return align_model, align_metadata


def merge_chunks(segments, chunk_size=CHUNK_LENGTH):
    """
    Merge VAD segments into larger segments of size ~CHUNK_LENGTH.
    """
    curr_start = 0
    curr_end = 0
    merged_segments = []
    seg_idxs = []
    speaker_idxs = []
    for sdx, seg in enumerate(segments):
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

    vad_segments_list = []
    vad_segments = vad_pipeline(audio)
    for speech_turn in vad_segments.get_timeline().support():
        vad_segments_list.append(Segment(speech_turn.start, speech_turn.end, "UNKNOWN"))
    # merge segments to approx 30s inputs to make whisper most appropraite
    vad_segments = merge_chunks(vad_segments_list)

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


def assign_word_speakers(diarize_df, result_segments, fill_nearest=False):

    for seg in result_segments:
        wdf = pd.DataFrame(seg['word-segments'])
        if len(wdf['start'].dropna()) == 0:
            wdf['start'] = seg['start']
            wdf['end'] = seg['end']
        speakers = []
        for wdx, wrow in wdf.iterrows():
            diarize_df['intersection'] = np.minimum(diarize_df['end'], wrow['end']) - np.maximum(diarize_df['start'], wrow['start'])
            diarize_df['union'] = np.maximum(diarize_df['end'], wrow['end']) - np.minimum(diarize_df['start'], wrow['start'])
            # remove no hit
            if not fill_nearest:
                dia_tmp = diarize_df[diarize_df['intersection'] > 0]
            else:
                dia_tmp = diarize_df
            if len(dia_tmp) == 0:
                speaker = None
            else:
                speaker = dia_tmp.sort_values("intersection", ascending=False).iloc[0][2]
            speakers.append(speaker)
        seg['word-segments']['speaker'] = speakers
        seg["speaker"] = pd.Series(speakers).value_counts().index[0]

    # create word level segments for .srt
    word_seg = []
    for seg in result_segments:
        wseg = pd.DataFrame(seg["word-segments"])
        for wdx, wrow in wseg.iterrows():
            if wrow["start"] is not None:
                speaker = wrow['speaker']
                if speaker is None or speaker == np.nan:
                    speaker = "UNKNOWN"
                word_seg.append(
                    {
                        "start": wrow["start"],
                        "end": wrow["end"],
                        "text": f"[{speaker}]: " + seg["text"][int(wrow["segment-text-start"]):int(wrow["segment-text-end"])]
                    }
                )

    # TODO: create segments but split words on new speaker

    return result_segments, word_seg

class Segment:
    def __init__(self, start, end, speaker=None):
        self.start = start
        self.end = end
        self.speaker = speaker


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
    parser.add_argument("--vad_input", default=None, type=str)
    # diarization params
    parser.add_argument("--diarize", action='store_true')
    parser.add_argument("--min_speakers", default=None, type=int)
    parser.add_argument("--max_speakers", default=None, type=int)
    # output save params
    parser.add_argument("--output_dir", "-o", type=str, default=".", help="directory to save the outputs")
    parser.add_argument("--output_type", default="all", choices=["all", "srt", "srt-word", "vtt", "txt", "tsv", "ass", "ass-char"], help="File type for desired output save")

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

    vad_filter: bool = args.pop("vad_filter")
    vad_input: bool = args.pop("vad_input")

    diarize: bool = args.pop("diarize")
    min_speakers: int = args.pop("min_speakers")
    max_speakers: int = args.pop("max_speakers")

    vad_pipeline = None
    if vad_input is not None:
        vad_input = pd.read_csv(vad_input, header=None, sep= " ")
    elif vad_filter:
        from pyannote.audio import Pipeline
        vad_pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")

    diarize_pipeline = None
    if diarize:
        from pyannote.audio import Pipeline
        diarize_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1")

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
            with open(os.path.join(output_dir, audio_basename + ".srt"), "w", encoding="utf-8") as srt:
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
        
        # save ASS character-level
        if output_type in ["ass-char", "all"]:
            with open(os.path.join(output_dir, audio_basename + ".char.ass"), "w", encoding="utf-8") as ass:
                write_ass(result_aligned["segments"], file=ass, resolution="char")


if __name__ == "__main__":
    cli()
