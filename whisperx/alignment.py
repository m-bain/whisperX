"""
Forced Alignment with Whisper
C. Max Bain
"""
from dataclasses import dataclass
from typing import Iterable, Optional, Union, List

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from whisperx.audio import SAMPLE_RATE, load_audio
from whisperx.utils import PUNKT_LANGUAGES, interpolate_nans
from whisperx.schema import (
    AlignedTranscriptionResult,
    CharAlignmentArrays,
    SingleSegment,
    SingleAlignedSegment,
    SegmentData,
    ProgressCallback,
)
import nltk
from nltk.data import load as nltk_load
from whisperx.log_utils import get_logger

logger = get_logger(__name__)

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
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "cs": "comodoro/wav2vec2-xls-r-300m-cs-250",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "fa": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "tr": "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish",
    "da": "saattrupdan/wav2vec2-xls-r-300m-ftspeech",
    "he": "imvladikon/wav2vec2-xls-r-300m-hebrew",
    "vi": 'nguyenvulebinh/wav2vec2-base-vi-vlsp2020',
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
    "ur": "kingabzpro/wav2vec2-large-xls-r-300m-Urdu",
    "te": "anuragshas/wav2vec2-large-xlsr-53-telugu",
    "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "ca": "softcatala/wav2vec2-large-xlsr-catala",
    "ml": "gvs/wav2vec2-large-xlsr-malayalam",
    "no": "NbAiLab/nb-wav2vec2-1b-bokmaal-v2",
    "nn": "NbAiLab/nb-wav2vec2-1b-nynorsk",
    "sk": "comodoro/wav2vec2-xls-r-300m-sk-cv8",
    "sl": "anton-l/wav2vec2-large-xlsr-53-slovenian",
    "hr": "classla/wav2vec2-xls-r-parlaspeech-hr",
    "ro": "gigant/romanian-wav2vec2",
    "eu": "stefan-it/wav2vec2-large-xlsr-53-basque",
    "gl": "ifrz/wav2vec2-large-xlsr-galician",
    "ka": "xsway/wav2vec2-large-xlsr-georgian",
    "lv": "jimregan/wav2vec2-large-xlsr-latvian-cv",
    "tl": "Khalsuu/filipino-wav2vec2-l-xls-r-300m-official",
    "sv": "KBLab/wav2vec2-large-voxrex-swedish",
    "id": "cahya/wav2vec2-large-xlsr-indonesian",
}


def _get_sentence_words(
    alignments: CharAlignmentArrays,
    start: int,
    stop: int,
) -> list[dict]:
    sentence_words = []
    word_start = start
    while word_start < stop:
        word_id = alignments.word_ids[word_start]
        word_stop = word_start + 1
        while word_stop < stop and alignments.word_ids[word_stop] == word_id:
            word_stop += 1

        word_text = "".join(alignments.chars[word_start:word_stop]).strip()
        if word_text:
            aligned_indices = [
                index
                for index in range(word_start, word_stop)
                if alignments.chars[index] != " "
            ]
            word_start_time = np.nanmin(alignments.starts[aligned_indices])
            word_end_time = np.nanmax(alignments.ends[aligned_indices])
            word_score = round(np.nanmean(alignments.scores[aligned_indices]), 3)

            word_segment = {"word": word_text}
            if not np.isnan(word_start_time):
                word_segment["start"] = word_start_time
            if not np.isnan(word_end_time):
                word_segment["end"] = word_end_time
            if not np.isnan(word_score):
                word_segment["score"] = word_score
            sentence_words.append(word_segment)
        word_start = word_stop
    return sentence_words


def _get_aligned_subsegments(
    alignments: CharAlignmentArrays,
    sentence_spans: list[tuple[int, int]],
    text: str,
    model_lang: str,
    interpolate_method: str,
    return_char_alignments: bool,
    avg_logprob: Optional[float],
) -> list[SingleAlignedSegment]:
    aligned_subsegments = []
    for sstart, send in sentence_spans:
        # Preserve the existing inclusive character selection at `send`,
        # while sentence text itself uses Punkt's exclusive end offset.
        char_stop = min(send + 1, len(alignments.chars))
        sentence_start = np.nanmin(alignments.starts[sstart:char_stop])
        non_space_indices = [
            index
            for index in range(sstart, char_stop)
            if alignments.chars[index] != " "
        ]
        sentence_end = np.nanmax(alignments.ends[non_space_indices])
        sentence_words = _get_sentence_words(alignments, sstart, char_stop)

        if sentence_words:
            word_starts = [word.get("start", np.nan) for word in sentence_words]
            word_ends = [word.get("end", np.nan) for word in sentence_words]
            if np.isnan(word_starts).any() and not np.isnan(word_starts).all():
                word_starts = interpolate_nans(word_starts, interpolate_method)
                word_ends = interpolate_nans(word_ends, interpolate_method)
                for index, word in enumerate(sentence_words):
                    if "start" not in word and not np.isnan(word_starts[index]):
                        word["start"] = float(word_starts[index])
                    if "end" not in word and not np.isnan(word_ends[index]):
                        word["end"] = float(word_ends[index])

        subsegment: SingleAlignedSegment = {
            "text": text[sstart:send],
            "start": sentence_start,
            "end": sentence_end,
            "words": sentence_words,
        }
        if avg_logprob is not None:
            subsegment["avg_logprob"] = avg_logprob
        if return_char_alignments:
            sentence_chars = []
            for index in range(sstart, char_stop):
                char = {"char": alignments.chars[index]}
                if not np.isnan(alignments.starts[index]):
                    char["start"] = float(alignments.starts[index])
                if not np.isnan(alignments.ends[index]):
                    char["end"] = float(alignments.ends[index])
                if not np.isnan(alignments.scores[index]):
                    char["score"] = float(alignments.scores[index])
                sentence_chars.append(char)
            subsegment["chars"] = sentence_chars
        aligned_subsegments.append(subsegment)

    starts = interpolate_nans(
        [segment["start"] for segment in aligned_subsegments],
        interpolate_method,
    )
    ends = interpolate_nans(
        [segment["end"] for segment in aligned_subsegments],
        interpolate_method,
    )

    separator = "" if model_lang in LANGUAGES_WITHOUT_SPACES else " "
    grouped: dict[tuple[float, float], SingleAlignedSegment] = {}
    for segment, start, end in zip(aligned_subsegments, starts, ends, strict=True):
        if np.isnan(start) or np.isnan(end):
            continue
        key = (float(start), float(end))
        if key not in grouped:
            segment["start"], segment["end"] = key
            grouped[key] = segment
            continue
        grouped_segment = grouped[key]
        grouped_segment["text"] += separator + segment["text"]
        grouped_segment["words"].extend(segment["words"])
        if return_char_alignments:
            grouped_segment["chars"].extend(segment["chars"])
    return [grouped[key] for key in sorted(grouped)]


def load_align_model(language_code: str, device: str, model_name: Optional[str] = None, model_dir=None, model_cache_only: bool = False):
    if model_name is None:
        # use default model
        if language_code in DEFAULT_ALIGN_MODELS_TORCH:
            model_name = DEFAULT_ALIGN_MODELS_TORCH[language_code]
        elif language_code in DEFAULT_ALIGN_MODELS_HF:
            model_name = DEFAULT_ALIGN_MODELS_HF[language_code]
        else:
            logger.error(f"No default alignment model for language: {language_code}. "
                         f"Please find a wav2vec2.0 model finetuned on this language at https://huggingface.co/models, "
                         f"then pass the model name via --align_model [MODEL_NAME]")
            raise ValueError(f"No default align-model for language: {language_code}")

    if model_name in torchaudio.pipelines.__all__:
        pipeline_type = "torchaudio"
        bundle = torchaudio.pipelines.__dict__[model_name]
        align_model = bundle.get_model(dl_kwargs={"model_dir": model_dir}).to(device)
        labels = bundle.get_labels()
        align_dictionary = {c.lower(): i for i, c in enumerate(labels)}
    else:
        try:
            processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=model_dir, local_files_only=model_cache_only)
            align_model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=model_dir, local_files_only=model_cache_only)
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


def _prepare_audio(audio: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    return audio


def _prepare_alignment(
    transcript: Iterable[SingleSegment],
    model_lang: str,
    model_dictionary: dict,
) -> dict[int, SegmentData]:
    punkt_lang = PUNKT_LANGUAGES.get(model_lang, "english")
    try:
        sentence_splitter = nltk_load(f"tokenizers/punkt_tab/{punkt_lang}.pickle")
    except LookupError as error:
        logger.info("Downloading NLTK punkt_tab data for sentence splitting...")
        if not nltk.download("punkt_tab", quiet=True):
            raise RuntimeError(
                "Failed to download NLTK 'punkt_tab' data, which is required for sentence splitting. "
                "Check your network connection, or install it manually with: python -m nltk.downloader punkt_tab"
            ) from error
        sentence_splitter = nltk_load(f"tokenizers/punkt_tab/{punkt_lang}.pickle")

    segment_data = {}
    for segment_index, segment in enumerate(transcript):
        text = segment["text"]
        num_leading = len(text) - len(text.lstrip())
        num_trailing = len(text) - len(text.rstrip())
        per_word = text if model_lang in LANGUAGES_WITHOUT_SPACES else text.split(" ")

        clean_char, clean_cdx = [], []
        for char_index, char in enumerate(text):
            normalized = char.lower()
            if model_lang not in LANGUAGES_WITHOUT_SPACES:
                normalized = normalized.replace(" ", "|")
            if char_index < num_leading or char_index > len(text) - num_trailing - 1:
                continue
            if normalized in model_dictionary or normalized not in (" ", "|"):
                clean_char.append(normalized)
                clean_cdx.append(char_index)

        segment_data[segment_index] = {
            "clean_char": clean_char,
            "clean_cdx": clean_cdx,
            "clean_wdx": list(range(len(per_word))),
            "sentence_spans": list(sentence_splitter.span_tokenize(text)),
        }
    return segment_data


def _compute_emission(
    waveform: torch.Tensor,
    model: torch.nn.Module,
    model_type: str,
    device: str,
) -> torch.Tensor:
    if waveform.shape[-1] < 400:
        lengths = torch.as_tensor([waveform.shape[-1]], device=device)
        waveform = torch.nn.functional.pad(waveform, (0, 400 - waveform.shape[-1]))
    else:
        lengths = None

    with torch.inference_mode():
        if model_type == "torchaudio":
            emissions, _ = model(waveform.to(device), lengths=lengths)
        elif model_type == "huggingface":
            emissions = model(waveform.to(device)).logits
        else:
            raise NotImplementedError(f"Align model of type {model_type} not supported.")
        return torch.log_softmax(emissions, dim=-1)[0].cpu().detach()


def _compute_emission_batch(
    waveforms: list[torch.Tensor],
    model: torch.nn.Module,
    model_type: str,
    device: str,
) -> list[torch.Tensor]:
    if not waveforms:
        return []

    flattened = []
    for waveform in waveforms:
        if waveform.ndim == 2 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)
        elif waveform.ndim != 1:
            raise ValueError(
                "Each waveform must have shape [time] or [1, time], "
                f"but found {list(waveform.shape)}."
            )
        flattened.append(waveform)

    with torch.inference_mode():
        if model_type == "torchaudio":
            # Padding before GroupNorm changes shorter inputs. Run the only
            # GroupNorm convolution per item, then batch the remaining model.
            first_block = model.feature_extractor.conv_layers[0]
            sequences = []
            for waveform in flattened:
                if waveform.shape[-1] < 400:
                    waveform = torch.nn.functional.pad(
                        waveform, (0, 400 - waveform.shape[-1])
                    )
                features, _ = first_block(
                    waveform.unsqueeze(0).unsqueeze(0).to(device), length=None
                )
                sequences.append(features.squeeze(0).transpose(0, 1))

            output_lengths = torch.as_tensor(
                [sequence.shape[0] for sequence in sequences],
                dtype=torch.long,
                device=device,
            )
            features = torch.nn.utils.rnn.pad_sequence(
                sequences, batch_first=True
            ).transpose(1, 2)
            for conv_layer in model.feature_extractor.conv_layers[1:]:
                features, output_lengths = conv_layer(features, output_lengths)
            emissions = model.encoder(features.transpose(1, 2), lengths=output_lengths)
            if model.aux is not None:
                emissions = model.aux(emissions)
        elif model_type == "huggingface":
            # Only LayerNorm models are invariant to padded batch inputs.
            if getattr(model.config, "feat_extract_norm", None) != "layer":
                return [
                    _compute_emission(waveform.unsqueeze(0), model, model_type, device)
                    for waveform in flattened
                ]
            lengths = torch.as_tensor(
                [waveform.shape[-1] for waveform in flattened],
                dtype=torch.long,
                device=device,
            )
            waveform_batch = torch.nn.utils.rnn.pad_sequence(
                flattened, batch_first=True
            )
            if waveform_batch.shape[-1] < 400:
                waveform_batch = torch.nn.functional.pad(
                    waveform_batch, (0, 400 - waveform_batch.shape[-1])
                )
            waveform_batch = waveform_batch.to(device)
            effective_lengths = lengths.clamp_min(400)
            attention_mask = (
                torch.arange(waveform_batch.shape[-1], device=device).unsqueeze(0)
                < effective_lengths.unsqueeze(1)
            ).long()
            emissions = model(waveform_batch, attention_mask=attention_mask).logits
            output_lengths = model._get_feat_extract_output_lengths(effective_lengths)
        else:
            raise NotImplementedError(f"Align model of type {model_type} not supported.")
        emissions = torch.log_softmax(emissions, dim=-1).cpu().detach()

    return [
        emission[:output_length]
        for emission, output_length in zip(
            emissions, output_lengths.cpu().tolist(), strict=True
        )
    ]


def _unaligned_segment(
    segment: SingleSegment,
    return_char_alignments: bool,
) -> SingleAlignedSegment:
    result: SingleAlignedSegment = {
        "start": segment["start"],
        "end": segment["end"],
        "text": segment["text"],
        "words": [],
        "chars": [] if return_char_alignments else None,
    }
    if "avg_logprob" in segment:
        result["avg_logprob"] = segment["avg_logprob"]
    return result


def _finish_alignment_segment(
    segment: SingleSegment,
    data: SegmentData,
    waveform: torch.Tensor,
    emission: torch.Tensor,
    model_dictionary: dict,
    model_lang: str,
    interpolate_method: str,
    return_char_alignments: bool,
) -> Optional[list[SingleAlignedSegment]]:
    text = segment["text"]
    text_clean = "".join(data["clean_char"])
    blank_id = next(
        (code for char, code in model_dictionary.items() if char in ("[pad]", "<pad>")),
        0,
    )

    has_wildcard = any(char not in model_dictionary for char in text_clean)
    if has_wildcard:
        non_blank_mask = torch.ones(emission.size(1), dtype=torch.bool)
        non_blank_mask[blank_id] = False
        wildcard = emission[:, non_blank_mask].max(dim=1).values
        emission = torch.cat([emission, wildcard.unsqueeze(1)], dim=1)
        tokens = [model_dictionary.get(char, emission.size(1) - 1) for char in text_clean]
    else:
        tokens = [model_dictionary[char] for char in text_clean]

    trellis = get_trellis(emission, tokens, blank_id)
    path = backtrack(trellis, emission, tokens, blank_id)
    if path is None:
        return None

    char_segments = merge_repeats(path, text_clean)
    ratio = (
        (segment["end"] - segment["start"])
        * waveform.size(0)
        / (trellis.size(0) - 1)
    )
    chars = list(text)
    starts = np.full(len(chars), np.nan, dtype=np.float64)
    ends = np.full(len(chars), np.nan, dtype=np.float64)
    scores = np.full(len(chars), np.nan, dtype=np.float64)
    word_ids = np.empty(len(chars), dtype=np.int32)
    for char_index, char_segment in zip(data["clean_cdx"], char_segments, strict=True):
        starts[char_index] = round(char_segment.start * ratio + segment["start"], 3)
        ends[char_index] = round(char_segment.end * ratio + segment["start"], 3)
        scores[char_index] = round(char_segment.score, 3)

    word_index = 0
    for char_index in range(len(chars)):
        word_ids[char_index] = word_index
        if model_lang in LANGUAGES_WITHOUT_SPACES:
            word_index += 1
        elif char_index == len(chars) - 1 or chars[char_index + 1] == " ":
            word_index += 1

    return _get_aligned_subsegments(
        CharAlignmentArrays(chars, starts, ends, scores, word_ids),
        data["sentence_spans"],
        text,
        model_lang,
        interpolate_method,
        return_char_alignments,
        segment.get("avg_logprob"),
    )


def _assemble_alignment_result(
    aligned_segments: list[SingleAlignedSegment],
) -> AlignedTranscriptionResult:
    return {
        "segments": aligned_segments,
        "word_segments": [
            word for segment in aligned_segments for word in segment["words"]
        ],
    }


def align(
    transcript: Iterable[SingleSegment],
    model: torch.nn.Module,
    align_model_metadata: dict,
    audio: Union[str, np.ndarray, torch.Tensor],
    device: str,
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
    print_progress: bool = False,
    combined_progress: bool = False,
    progress_callback: ProgressCallback = None,
) -> AlignedTranscriptionResult:
    """
    Align phoneme recognition predictions to known transcription.
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

    # Use language-specific Punkt model if available otherwise we fallback to English.
    punkt_lang = PUNKT_LANGUAGES.get(model_lang, 'english')
    try:
        sentence_splitter = nltk_load(f'tokenizers/punkt_tab/{punkt_lang}.pickle')
    except LookupError as e:
        logger.info("Downloading NLTK punkt_tab data for sentence splitting...")
        if not nltk.download('punkt_tab', quiet=True):
            raise RuntimeError(
                "Failed to download NLTK 'punkt_tab' data, which is required for sentence splitting. "
                "Check your network connection, or install it manually with: python -m nltk.downloader punkt_tab"
            ) from e
        sentence_splitter = nltk_load(f'tokenizers/punkt_tab/{punkt_lang}.pickle')

    # 1. Preprocess to keep only characters in dictionary
    total_segments = len(transcript)
    # Store temporary processing values
    segment_data: dict[int, SegmentData] = {}
    for sdx, segment in enumerate(transcript):
        # strip spaces at beginning / end, but keep track of the amount.
        if print_progress:
            base_progress = ((sdx + 1) / total_segments) * 100
            percent_complete = (50 + base_progress / 2) if combined_progress else base_progress
            print(f"Progress: {percent_complete:.2f}%...")

        num_leading = len(segment["text"]) - len(segment["text"].lstrip())
        num_trailing = len(segment["text"]) - len(segment["text"].rstrip())
        text = segment["text"]

        # split into words
        if model_lang not in LANGUAGES_WITHOUT_SPACES:
            per_word = text.split(" ")
        else:
            per_word = text

        clean_char, clean_cdx = [], []
        for cdx, char in enumerate(text):
            char_ = char.lower()
            # wav2vec2 models use "|" character to represent spaces
            if model_lang not in LANGUAGES_WITHOUT_SPACES:
                char_ = char_.replace(" ", "|")

            # ignore whitespace at beginning and end of transcript
            if cdx < num_leading:
                pass
            elif cdx > len(text) - num_trailing - 1:
                pass
            elif char_ in model_dictionary.keys():
                clean_char.append(char_)
                clean_cdx.append(cdx)
            elif char_ not in (" ", "|"):
                # unknown char (digit, symbol, foreign script) — use wildcard
                clean_char.append(char_)
                clean_cdx.append(cdx)

        clean_wdx = list(range(len(per_word)))

        sentence_spans = list(sentence_splitter.span_tokenize(text))

        segment_data[sdx] = {
            "clean_char": clean_char,
            "clean_cdx": clean_cdx,
            "clean_wdx": clean_wdx,
            "sentence_spans": sentence_spans
        }

    aligned_segments: List[SingleAlignedSegment] = []

    # 2. Get prediction matrix from alignment model & align
    for sdx, segment in enumerate(transcript):

        t1 = segment["start"]
        t2 = segment["end"]
        text = segment["text"]
        avg_logprob = segment.get("avg_logprob")

        aligned_seg: SingleAlignedSegment = {
            "start": t1,
            "end": t2,
            "text": text,
            "words": [],
            "chars": None,
        }

        if avg_logprob is not None:
            aligned_seg["avg_logprob"] = avg_logprob

        if return_char_alignments:
            aligned_seg["chars"] = []

        # check we can align
        if len(segment_data[sdx]["clean_char"]) == 0:
            logger.warning(f'Failed to align segment ("{segment["text"]}"): no characters in this segment found in model dictionary, resorting to original')
            aligned_segments.append(aligned_seg)
            continue

        if t1 >= MAX_DURATION:
            logger.warning(f'Failed to align segment ("{segment["text"]}"): original start time longer than audio duration, skipping')
            aligned_segments.append(aligned_seg)
            continue

        f1 = int(t1 * SAMPLE_RATE)
        f2 = int(t2 * SAMPLE_RATE)

        waveform_segment = audio[:, f1:f2]
        emission = _compute_emission(waveform_segment, model, model_type, device)
        aligned_subsegments = _finish_alignment_segment(
            segment,
            segment_data[sdx],
            waveform_segment,
            emission,
            model_dictionary,
            model_lang,
            interpolate_method,
            return_char_alignments,
        )
        if aligned_subsegments is None:
            logger.warning(f'Failed to align segment ("{segment["text"]}"): backtrack failed, resorting to original')
            aligned_segments.append(aligned_seg)
            continue
        if progress_callback is not None:
            progress_callback(((sdx + 1) / total_segments) * 100)

        aligned_segments += aligned_subsegments

    return _assemble_alignment_result(aligned_segments)


def align_batch(
    transcripts: list[list[SingleSegment]],
    model: torch.nn.Module,
    align_model_metadata: dict,
    audio: list[Union[str, np.ndarray, torch.Tensor]],
    device: str,
    batch_size: int,
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
    print_progress: bool = False,
    combined_progress: bool = False,
    progress_callback: ProgressCallback = None,
) -> list[AlignedTranscriptionResult]:
    """Align multiple audio inputs with duration-sorted model batches."""
    if len(transcripts) != len(audio):
        raise ValueError(
            "transcripts and audio must contain the same number of items, "
            f"but found {len(transcripts)} transcripts and {len(audio)} audio items."
        )
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, but found {batch_size}.")

    audios = [_prepare_audio(audio_item) for audio_item in audio]
    model_dictionary = align_model_metadata["dictionary"]
    model_lang = align_model_metadata["language"]
    model_type = align_model_metadata["type"]
    segment_results: list[list[Optional[list[SingleAlignedSegment]]]] = [
        [None] * len(transcript) for transcript in transcripts
    ]
    jobs = []
    total_segments = sum(len(transcript) for transcript in transcripts)
    processed_segments = 0

    for audio_index, (transcript, audio_item) in enumerate(
        zip(transcripts, audios, strict=True)
    ):
        max_duration = audio_item.shape[1] / SAMPLE_RATE
        segment_data = _prepare_alignment(transcript, model_lang, model_dictionary)
        for segment_index, segment in enumerate(transcript):
            processed_segments += 1
            if print_progress and total_segments:
                base_progress = processed_segments / total_segments * 100
                percent_complete = (
                    50 + base_progress / 2 if combined_progress else base_progress
                )
                print(f"Progress: {percent_complete:.2f}%...")

            fallback = _unaligned_segment(segment, return_char_alignments)
            if not segment_data[segment_index]["clean_char"]:
                logger.warning(
                    f'Failed to align segment ("{segment["text"]}"): no characters '
                    "in this segment found in model dictionary, resorting to original"
                )
                segment_results[audio_index][segment_index] = [fallback]
                continue
            if segment["start"] >= max_duration:
                logger.warning(
                    f'Failed to align segment ("{segment["text"]}"): original start '
                    "time longer than audio duration, skipping"
                )
                segment_results[audio_index][segment_index] = [fallback]
                continue

            first_sample = int(segment["start"] * SAMPLE_RATE)
            last_sample = int(segment["end"] * SAMPLE_RATE)
            jobs.append(
                {
                    "audio_index": audio_index,
                    "segment_index": segment_index,
                    "segment": segment,
                    "data": segment_data[segment_index],
                    "waveform": audio_item[:, first_sample:last_sample],
                    "fallback": fallback,
                }
            )

    jobs.sort(key=lambda job: job["waveform"].shape[-1])
    completed_segments = 0
    for batch_start in range(0, len(jobs), batch_size):
        batch = jobs[batch_start:batch_start + batch_size]
        emissions = _compute_emission_batch(
            [job["waveform"] for job in batch], model, model_type, device
        )
        if len(emissions) != len(batch):
            raise RuntimeError(
                "Alignment model returned an unexpected number of emissions: "
                f"expected {len(batch)}, found {len(emissions)}."
            )
        for job, emission in zip(batch, emissions, strict=True):
            aligned = _finish_alignment_segment(
                job["segment"],
                job["data"],
                job["waveform"],
                emission,
                model_dictionary,
                model_lang,
                interpolate_method,
                return_char_alignments,
            )
            if aligned is None:
                logger.warning(
                    f'Failed to align segment ("{job["segment"]["text"]}"): '
                    "backtrack failed, resorting to original"
                )
                aligned = [job["fallback"]]
            segment_results[job["audio_index"]][job["segment_index"]] = aligned
            completed_segments += 1
            if progress_callback is not None and total_segments:
                progress_callback(completed_segments / total_segments * 100)

    results = []
    for audio_results in segment_results:
        aligned_segments = []
        for aligned in audio_results:
            if aligned is None:
                raise RuntimeError("Internal error: alignment segment was not processed.")
            aligned_segments.extend(aligned)
        results.append(_assemble_alignment_result(aligned_segments))
    return results


"""
source: https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html
"""


def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra dimensions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, blank_id], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else blank_id].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        # failed
        return None

    return path[::-1]


# Merge the labels
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start

def merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments

def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words
