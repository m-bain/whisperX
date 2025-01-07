""""
Forced Alignment with Whisper
C. Max Bain
"""
from dataclasses import dataclass
from multiprocessing.pool import AsyncResult
import time
import traceback
from typing import Iterable, Optional, Union, List

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.multiprocessing.pool import Pool
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from .audio import SAMPLE_RATE, load_audio
from .utils import interpolate_nans
from .types import AlignedTranscriptionResult, SingleSegment, SingleAlignedSegment, SingleWordSegment
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

PUNKT_ABBREVIATIONS = ['dr', 'vs', 'mr', 'mrs', 'prof']

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
    "vi": 'nguyenvulebinh/wav2vec2-base-vi',
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
}


def load_align_model(language_code: str, device: str, model_name: Optional[str] = None, model_dir=None):
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
        align_model = bundle.get_model(dl_kwargs={"model_dir": model_dir}).to(device)
        labels = bundle.get_labels()
        align_dictionary = {c.lower(): i for i, c in enumerate(labels)}
    else:
        try:
            processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=model_dir)
            align_model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=model_dir)
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

def align_backtrack(
    transcript: Iterable[SingleSegment],
    model_lang: str,
    audio: torch.Tensor,
    emission_list: List[torch.Tensor],
    blank_id_list: List[int],
    tokens_list: List[List[int]],
    waveform_segment_sizes_list: List[int],
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
    debug_id: Optional[str] = None,
) -> AlignedTranscriptionResult:
    """
    Align phoneme recognition predictions to known transcription.
    """

    try:
        MAX_DURATION = audio.shape[1] / SAMPLE_RATE

        aligned_segments: List[SingleAlignedSegment] = []
        assert len(tokens_list) == len(emission_list) == len(blank_id_list) == len(waveform_segment_sizes_list)

        for sdx, segment in enumerate(transcript):
            t1 = segment["start"]
            t2 = segment["end"]

            emission = emission_list[sdx]
            # convert emission to numpy
            emission = emission.numpy()
            blank_id = blank_id_list[sdx]
            tokens = tokens_list[sdx]
            waveform_segment_size = waveform_segment_sizes_list[sdx]
            text = segment["text"]

            aligned_seg: SingleAlignedSegment = {
                "start": t1,
                "end": t2,
                "text": text,
                "words": [],
            }

            if return_char_alignments:
                aligned_seg["chars"] = []

            # check we can align
            if len(segment["clean_char"]) == 0:
                print(f'Failed to align segment ("{segment["text"]}"): no characters in this segment found in model dictionary, resorting to original...')
                aligned_segments.append(aligned_seg)
                continue

            if t1 >= MAX_DURATION:
                print(f'Failed to align segment ("{segment["text"]}"): original start time longer than audio duration, skipping...')
                aligned_segments.append(aligned_seg)
                continue

            text_clean = "".join(segment["clean_char"])

            print("getting trellis for sample", debug_id)
            start_time_trellis = time.time()

            trellis = get_trellis(emission, tokens, blank_id)
            print("doing backtrack for sample", debug_id, ". Trellis took", time.time() - start_time_trellis)
            path = backtrack(trellis, emission, tokens, blank_id)

            if path is None:
                print(f'Failed to align segment ("{segment["text"]}"): backtrack failed, resorting to original...')
                aligned_segments.append(aligned_seg)
                continue

            char_segments = merge_repeats(path, text_clean)

            duration = t2 - t1
            trellis_size = trellis.shape[0]
            ratio = duration * waveform_segment_size / (trellis_size - 1)

            # assign timestamps to aligned characters
            char_segments_arr = []
            word_idx = 0
            for cdx, char in enumerate(text):
                start, end, score = None, None, None
                if cdx in segment["clean_cdx"]:
                    char_seg = char_segments[segment["clean_cdx"].index(cdx)]
                    start = round(char_seg.start * ratio + t1, 3)
                    end = round(char_seg.end * ratio + t1, 3)
                    score = round(char_seg.score, 3)

                char_segments_arr.append(
                    {
                        "char": char,
                        "start": start,
                        "end": end,
                        "score": score,
                        "word-idx": word_idx,
                    }
                )

                # increment word_idx, nltk word tokenization would probably be more robust here, but us space for now...
                if model_lang in LANGUAGES_WITHOUT_SPACES:
                    word_idx += 1
                elif cdx == len(text) - 1 or text[cdx+1] == " ":
                    word_idx += 1

            char_segments_arr = pd.DataFrame(char_segments_arr)

            aligned_subsegments = []
            # assign sentence_idx to each character index
            char_segments_arr["sentence-idx"] = None
            for sdx, (sstart, send) in enumerate(segment["sentence_spans"]):
                curr_chars = char_segments_arr.loc[(char_segments_arr.index >= sstart) & (char_segments_arr.index <= send)]
                char_segments_arr.loc[(char_segments_arr.index >= sstart) & (char_segments_arr.index <= send), "sentence-idx"] = sdx

                sentence_text = text[sstart:send]
                sentence_start = curr_chars["start"].min()
                end_chars = curr_chars[curr_chars["char"] != ' ']
                sentence_end = end_chars["end"].max()
                sentence_words = []

                for word_idx in curr_chars["word-idx"].unique():
                    word_chars = curr_chars.loc[curr_chars["word-idx"] == word_idx]
                    word_text = "".join(word_chars["char"].tolist()).strip()
                    if len(word_text) == 0:
                        continue

                    # dont use space character for alignment
                    word_chars = word_chars[word_chars["char"] != " "]

                    word_start = word_chars["start"].min()
                    word_end = word_chars["end"].max()
                    word_score = round(word_chars["score"].mean(), 3)

                    # -1 indicates unalignable
                    word_segment = {"word": word_text}

                    if not np.isnan(word_start):
                        word_segment["start"] = word_start
                    if not np.isnan(word_end):
                        word_segment["end"] = word_end
                    if not np.isnan(word_score):
                        word_segment["score"] = word_score

                    sentence_words.append(word_segment)

                aligned_subsegments.append({
                    "text": sentence_text,
                    "start": sentence_start,
                    "end": sentence_end,
                    "words": sentence_words,
                })

                if return_char_alignments:
                    curr_chars = curr_chars[["char", "start", "end", "score"]]
                    curr_chars.fillna(-1, inplace=True)
                    curr_chars = curr_chars.to_dict("records")
                    curr_chars = [{key: val for key, val in char.items() if val != -1} for char in curr_chars]
                    aligned_subsegments[-1]["chars"] = curr_chars

            aligned_subsegments = pd.DataFrame(aligned_subsegments)
            aligned_subsegments["start"] = interpolate_nans(aligned_subsegments["start"], method=interpolate_method)
            aligned_subsegments["end"] = interpolate_nans(aligned_subsegments["end"], method=interpolate_method)
            # concatenate sentences with same timestamps
            agg_dict = {"text": " ".join, "words": "sum"}
            if model_lang in LANGUAGES_WITHOUT_SPACES:
                agg_dict["text"] = "".join
            if return_char_alignments:
                agg_dict["chars"] = "sum"
            aligned_subsegments= aligned_subsegments.groupby(["start", "end"], as_index=False).agg(agg_dict)
            aligned_subsegments = aligned_subsegments.to_dict('records')
            aligned_segments += aligned_subsegments

        # create word_segments list
        word_segments: List[SingleWordSegment] = []
        for segment in aligned_segments:
            word_segments += segment["words"]
    except Exception as e:
        print(f"Error in align_backtrack: {e}")
        traceback.print_exc()
        raise
    print(f"Backtracking done for sample {debug_id}")
    return {"segments": aligned_segments, "word_segments": word_segments}

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
    pool: Pool | None = None,
    debug_id: str | None = None,
) -> AsyncResult:
    """
    Align phoneme recognition predictions to known transcription.
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)

    model_dictionary = align_model_metadata["dictionary"]
    model_lang = align_model_metadata["language"]
    model_type = align_model_metadata["type"]

    # 1. Preprocess to keep only characters in dictionary
    total_segments = len(transcript)
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

        clean_wdx = []
        for wdx, wrd in enumerate(per_word):
            if any([c in model_dictionary.keys() for c in wrd]):
                clean_wdx.append(wdx)


        punkt_param = PunktParameters()
        punkt_param.abbrev_types = set(PUNKT_ABBREVIATIONS)
        sentence_splitter = PunktSentenceTokenizer(punkt_param)
        sentence_spans = list(sentence_splitter.span_tokenize(text))

        segment["clean_char"] = clean_char
        segment["clean_cdx"] = clean_cdx
        segment["clean_wdx"] = clean_wdx
        segment["sentence_spans"] = sentence_spans


    # 2. Get prediction matrix from alignment model & align

    emission_list = []
    blank_id_list = []
    tokens_list = []
    waveform_segment_sizes_list = []
    for sdx, segment in enumerate(transcript):

        t1 = segment["start"]
        t2 = segment["end"]

        f1 = int(t1 * SAMPLE_RATE)
        f2 = int(t2 * SAMPLE_RATE)

        # TODO: Probably can get some speedup gain with batched inference here
        waveform_segment = audio[:, f1:f2]
        # Handle the minimum input length for wav2vec2 models
        if waveform_segment.shape[-1] < 400:
            lengths = torch.as_tensor([waveform_segment.shape[-1]]).to(device)
            waveform_segment = torch.nn.functional.pad(
                waveform_segment, (0, 400 - waveform_segment.shape[-1])
            )
        else:
            lengths = None

        with torch.inference_mode():
            if model_type == "torchaudio":
                emissions, _ = model(waveform_segment.to(device), lengths=lengths)
            elif model_type == "huggingface":
                emissions = model(waveform_segment.to(device)).logits
            else:
                raise NotImplementedError(f"Align model of type {model_type} not supported.")
            emissions = torch.log_softmax(emissions, dim=-1)

        emission_list.append(emissions[0].cpu().detach())
        blank_id = 0
        for char, code in model_dictionary.items():
            if char == '[pad]' or char == '<pad>':
                blank_id = code
        blank_id_list.append(blank_id)

        text_clean = "".join(segment["clean_char"])
        tokens = [model_dictionary[c] for c in text_clean]
        tokens_list.append(tokens)
        waveform_segment_sizes_list.append(waveform_segment.size(0))
    print(f"Sample {debug_id} done aligning, time to backtrack")
    # we need emission array, tokens, and blank_ids
    # waveform_segment.size(),
    return pool.apply_async(align_backtrack, (
        transcript,
        model_lang,
        audio,
        emission_list,
        blank_id_list,
        tokens_list,
        waveform_segment_sizes_list,
        interpolate_method,
        return_char_alignments,
        debug_id,
    ))

"""
source: https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html
"""
def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.shape[0]
    num_tokens = len(tokens)

    # Initialize trellis with the necessary dimensions
    trellis = np.empty((num_frame + 1, num_tokens + 1), dtype=np.float64)
    trellis[0, 0] = 0
    # Initialize the first column of the trellis
    trellis[1:, 0] = np.cumsum(emission[:, 0], axis=0)
    trellis[0, -num_tokens:] = -np.inf
    trellis[-num_tokens:, 0] = np.inf

    for t in range(num_frame):
        trellis[t + 1, 1:] = np.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens]
        )
    return trellis

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

def backtrack(trellis, emission, tokens, blank_id=0):
    """
    Backtrack through a trellis matrix to find the optimal path.

    Args:
        trellis: numpy array of shape (T, J)
        emission: numpy array of shape (T, V) where V is vocab size
        tokens: list or array of token indices
        blank_id: ID of the blank token (default: 0)

    Returns:
        list of Points or None if backtracking fails
    """
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.shape[1] - 1
    t_start = np.argmax(trellis[:, j])
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
        # Note: np.exp is used instead of .exp() method
        prob = np.exp(emission[t - 1, tokens[j - 1] if changed > stayed else 0])
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, float(prob)))  # Convert to float in case it's numpy.float32
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
