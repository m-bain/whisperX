""""
Forced Alignment with Whisper
C. Max Bain
"""
import numpy as np
import pandas as pd
from typing import List, Union, Iterator, TYPE_CHECKING
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
import torch
from dataclasses import dataclass
from whisper.audio import SAMPLE_RATE, load_audio
from .utils import interpolate_nans


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
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "fa": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "tr": "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish",
}


def load_align_model(language_code, device, model_name=None, model_dir=None):
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
            processor = Wav2Vec2Processor.from_pretrained(model_name)
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

    diarization: pd.DataFrame {'start': List[float], 'end': List[float], 'speaker': List[float]}
        diarization segments with speaker labels.

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

    char_segments_arr = {
        "segment-idx": [],
        "subsegment-idx": [],
        "word-idx": [],
        "char": [],
        "start": [],
        "end": [],
        "score": [],
    }

    for sdx, segment in enumerate(transcript):
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
                print(f'Failed to align segment ("{segment["text"]}"): no characters in this segment found in model dictionary, resorting to original...')
                break          
           
            transcription_cleaned = "".join(clean_char)
            tokens = [model_dictionary[c] for c in transcription_cleaned]

            # we only pad if not using VAD filtering
            if "seg_text" not in segment:
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
                print(f'Failed to align segment ("{segment["text"]}"): backtrack failed, resorting to original...')
                break
            char_segments = merge_repeats(path, transcription_cleaned)
            # word_segments = merge_words(char_segments)
            

            # sub-segments
            if "seg-text" not in segment:
                segment["seg-text"] = [transcription]
                
            seg_lens = [0] + [len(x) for x in segment["seg-text"]]
            seg_lens_cumsum = list(np.cumsum(seg_lens))
            sub_seg_idx = 0

            wdx = 0
            duration = t2 - t1
            ratio = duration * waveform_segment.size(0) / (trellis.size(0) - 1)
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

                char_segments_arr["char"].append(char)
                char_segments_arr["start"].append(start)
                char_segments_arr["end"].append(end)
                char_segments_arr["score"].append(score)
                char_segments_arr["word-idx"].append(wdx)
                char_segments_arr["segment-idx"].append(sdx)
                char_segments_arr["subsegment-idx"].append(sub_seg_idx)

                # word-level info
                if model_lang in LANGUAGES_WITHOUT_SPACES:
                    # character == word
                    wdx += 1
                elif is_last or transcription[cdx+1] == " " or cdx == seg_lens_cumsum[sub_seg_idx+1] - 1:
                    wdx += 1

                if is_last or cdx == seg_lens_cumsum[sub_seg_idx+1] - 1:
                    wdx = 0
                    sub_seg_idx += 1

            prev_t2 = segment["end"]

            segment_align_success = True
            # end while True loop
            break

        # reset prev_t2 due to drifting issues
        if not segment_align_success:
            prev_t2 = 0
        
    char_segments_arr = pd.DataFrame(char_segments_arr)
    not_space = char_segments_arr["char"] != " "

    per_seg_grp = char_segments_arr.groupby(["segment-idx", "subsegment-idx"], as_index = False)
    char_segments_arr = per_seg_grp.apply(lambda x: x.reset_index(drop = True)).reset_index()
    per_word_grp = char_segments_arr[not_space].groupby(["segment-idx", "subsegment-idx", "word-idx"])
    per_subseg_grp = char_segments_arr[not_space].groupby(["segment-idx", "subsegment-idx"])
    per_seg_grp = char_segments_arr[not_space].groupby(["segment-idx"])
    char_segments_arr["local-char-idx"] = char_segments_arr.groupby(["segment-idx", "subsegment-idx"]).cumcount()
    per_word_grp = char_segments_arr[not_space].groupby(["segment-idx", "subsegment-idx", "word-idx"]) # regroup

    word_segments_arr = {}

    # start of word is first char with a timestamp
    word_segments_arr["start"] = per_word_grp["start"].min().values
    # end of word is last char with a timestamp
    word_segments_arr["end"] = per_word_grp["end"].max().values
    # score of word is mean (excluding nan)
    word_segments_arr["score"] = per_word_grp["score"].mean().values

    word_segments_arr["segment-text-start"] = per_word_grp["local-char-idx"].min().astype(int).values
    word_segments_arr["segment-text-end"] = per_word_grp["local-char-idx"].max().astype(int).values+1
    word_segments_arr = pd.DataFrame(word_segments_arr)

    word_segments_arr[["segment-idx", "subsegment-idx", "word-idx"]] = per_word_grp["local-char-idx"].min().reset_index()[["segment-idx", "subsegment-idx", "word-idx"]].astype(int)
    segments_arr = {}
    segments_arr["start"] = per_subseg_grp["start"].min().reset_index()["start"]
    segments_arr["end"] = per_subseg_grp["end"].max().reset_index()["end"]
    segments_arr = pd.DataFrame(segments_arr)
    segments_arr[["segment-idx", "subsegment-idx-start"]] = per_subseg_grp["start"].min().reset_index()[["segment-idx", "subsegment-idx"]]
    segments_arr["subsegment-idx-end"] = segments_arr["subsegment-idx-start"] + 1

    # interpolate missing words / sub-segments
    if interpolate_method != "ignore":
        wrd_subseg_grp = word_segments_arr.groupby(["segment-idx", "subsegment-idx"], group_keys=False)
        wrd_seg_grp = word_segments_arr.groupby(["segment-idx"], group_keys=False)
        # we still know which word timestamps are interpolated because their score == nan
        word_segments_arr["start"] = wrd_subseg_grp['start'].apply(lambda group: interpolate_nans(group, method=interpolate_method))
        word_segments_arr["end"] = wrd_subseg_grp['end'].apply(lambda group: interpolate_nans(group, method=interpolate_method))

        word_segments_arr["start"] = wrd_seg_grp['start'].apply(lambda group: interpolate_nans(group, method=interpolate_method))
        word_segments_arr["end"] = wrd_seg_grp['end'].apply(lambda group: interpolate_nans(group, method=interpolate_method))

        sub_seg_grp =  segments_arr.groupby(["segment-idx"], group_keys=False)
        segments_arr['start'] = sub_seg_grp['start'].apply(lambda group: interpolate_nans(group, method=interpolate_method))
        segments_arr['end'] = sub_seg_grp['end'].apply(lambda group: interpolate_nans(group, method=interpolate_method))

        # merge words & subsegments which are missing times
        word_grp = word_segments_arr.groupby(["segment-idx", "subsegment-idx", "end"])

        word_segments_arr["segment-text-start"] = word_grp["segment-text-start"].transform(min)
        word_segments_arr["segment-text-end"] = word_grp["segment-text-end"].transform(max)
        word_segments_arr.drop_duplicates(subset=["segment-idx", "subsegment-idx", "end"], inplace=True)

        seg_grp_dup = segments_arr.groupby(["segment-idx", "start", "end"])
        segments_arr["subsegment-idx-start"] = seg_grp_dup["subsegment-idx-start"].transform(min)
        segments_arr["subsegment-idx-end"] = seg_grp_dup["subsegment-idx-end"].transform(max)
        segments_arr.drop_duplicates(subset=["segment-idx", "subsegment-idx-start", "subsegment-idx-end"], inplace=True)
    else:
        word_segments_arr.dropna(inplace=True)
        segments_arr.dropna(inplace=True)

    # if some segments still have missing timestamps (usually because all numerals / symbols), then use original timestamps...
    segments_arr['start'].fillna(pd.Series([x['start'] for x in transcript]), inplace=True)
    segments_arr['end'].fillna(pd.Series([x['end'] for x in transcript]), inplace=True)
    segments_arr['subsegment-idx-start'].fillna(0, inplace=True)
    segments_arr['subsegment-idx-end'].fillna(1, inplace=True)


    aligned_segments = []
    aligned_segments_word = []

    word_segments_arr.set_index(["segment-idx", "subsegment-idx"], inplace=True)
    char_segments_arr.set_index(["segment-idx", "subsegment-idx", "word-idx"], inplace=True)

    for sdx, srow in segments_arr.iterrows():

        seg_idx = int(srow["segment-idx"])
        sub_start = int(srow["subsegment-idx-start"])
        sub_end = int(srow["subsegment-idx-end"])

        seg = transcript[seg_idx]
        text = "".join(seg["seg-text"][sub_start:sub_end])

        wseg = word_segments_arr.loc[seg_idx].loc[sub_start:sub_end-1]
        wseg["start"].fillna(srow["start"], inplace=True)
        wseg["end"].fillna(srow["end"], inplace=True)
        wseg["segment-text-start"].fillna(0, inplace=True)
        wseg["segment-text-end"].fillna(len(text)-1, inplace=True)

        cseg = char_segments_arr.loc[seg_idx].loc[sub_start:sub_end-1]
        # fixes bug for single segment in transcript
        cseg['segment-text-start'] = cseg['level_1'] if 'level_1' in cseg else 0
        cseg['segment-text-end'] = cseg['level_1'] + 1 if 'level_1' in cseg else 1
        if 'level_1' in cseg: del cseg['level_1']
        if 'level_0' in cseg: del cseg['level_0']
        cseg.reset_index(inplace=True)
        aligned_segments.append(
            {
                "start": srow["start"],
                "end": srow["end"],
                "text": text,
                "word-segments": wseg,
                "char-segments": cseg
            }
        )

        def get_raw_text(word_row):
            return seg["seg-text"][word_row.name][int(word_row["segment-text-start"]):int(word_row["segment-text-end"])+1]

        wdx = 0
        curr_text = get_raw_text(wseg.iloc[wdx])
        if len(wseg) > 1:
            for _, wrow in wseg.iloc[1:].iterrows():
                if wrow['start'] != wseg.iloc[wdx]['start']:
                    aligned_segments_word.append(
                        {
                            "text": curr_text.strip(),
                            "start": wseg.iloc[wdx]["start"],
                            "end": wseg.iloc[wdx]["end"],
                        }
                    )
                    curr_text = ""
                curr_text += " " + get_raw_text(wrow)
                wdx += 1
        aligned_segments_word.append(
            {
                "text": curr_text.strip(),
                "start": wseg.iloc[wdx]["start"],
                "end": wseg.iloc[wdx]["end"]
            }
        )

    
    return {"segments": aligned_segments, "word_segments": aligned_segments_word}


"""
source: https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html
"""
def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
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
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
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
