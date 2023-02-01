import os
import zlib
from typing import Callable, TextIO, Iterator, Tuple
import pandas as pd
import numpy as np

def exact_div(x, y):
    assert x % y == 0
    return x // y


def str2bool(string):
    str2val = {"True": True, "False": False}
    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")


def optional_int(string):
    return None if string == "None" else int(string)


def optional_float(string):
    return None if string == "None" else float(string)


def compression_ratio(text) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = '.'):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"


def write_txt(transcript: Iterator[dict], file: TextIO):
    for segment in transcript:
        print(segment['text'].strip(), file=file, flush=True)


def write_vtt(transcript: Iterator[dict], file: TextIO):
    print("WEBVTT\n", file=file)
    for segment in transcript:
        print(
            f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            f"{segment['text'].strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )

def write_tsv(transcript: Iterator[dict], file: TextIO):
    print("start", "end", "text", sep="\t", file=file)
    for segment in transcript:
        print(segment['start'], file=file, end="\t")
        print(segment['end'], file=file, end="\t")
        print(segment['text'].strip().replace("\t", " "), file=file, flush=True)


def write_srt(transcript: Iterator[dict], file: TextIO):
    """
    Write a transcript to a file in SRT format.

    Example usage:
        from pathlib import Path
        from whisper.utils import write_srt

        result = transcribe(model, audio_path, temperature=temperature, **args)

        # save SRT
        audio_basename = Path(audio_path).stem
        with open(Path(output_dir) / (audio_basename + ".srt"), "w", encoding="utf-8") as srt:
            write_srt(result["segments"], file=srt)
    """
    for i, segment in enumerate(transcript, start=1):
        # write srt lines
        print(
            f"{i}\n"
            f"{format_timestamp(segment['start'], always_include_hours=True, decimal_marker=',')} --> "
            f"{format_timestamp(segment['end'], always_include_hours=True, decimal_marker=',')}\n"
            f"{segment['text'].strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )


def write_ass(transcript: Iterator[dict],
            file: TextIO,
            resolution: str = "word",
            color: str = None, underline=True,
            prefmt: str = None, suffmt: str = None,
            font: str = None, font_size: int = 24,
            strip=True, **kwargs):
    """
    Credit: https://github.com/jianfch/stable-ts/blob/ff79549bd01f764427879f07ecd626c46a9a430a/stable_whisper/text_output.py
        Generate Advanced SubStation Alpha (ass) file from results to
    display both phrase-level & word-level timestamp simultaneously by:
     -using segment-level timestamps display phrases as usual
     -using word-level timestamps change formats (e.g. color/underline) of the word in the displayed segment
    Note: ass file is used in the same way as srt, vtt, etc.
    Parameters
    ----------
    transcript: dict
        results from modified model
    file: TextIO
        file object to write to
    resolution: str
        "word" or "char", timestamp resolution to highlight.
    color: str
        color code for a word at its corresponding timestamp
        <bbggrr> reverse order hexadecimal RGB value (e.g. FF0000 is full intensity blue. Default: 00FF00)
    underline: bool
        whether to underline a word at its corresponding timestamp
    prefmt: str
        used to specify format for word-level timestamps (must be use with 'suffmt' and overrides 'color'&'underline')
        appears as such in the .ass file:
            Hi, {<prefmt>}how{<suffmt>} are you?
        reference [Appendix A: Style override codes] in http://www.tcax.org/docs/ass-specs.htm
    suffmt: str
        used to specify format for word-level timestamps (must be use with 'prefmt' and overrides 'color'&'underline')
        appears as such in the .ass file:
            Hi, {<prefmt>}how{<suffmt>} are you?
        reference [Appendix A: Style override codes] in http://www.tcax.org/docs/ass-specs.htm
    font: str
        word font (default: Arial)
    font_size: int
        word font size (default: 48)
    kwargs:
        used for format styles:
        'Name', 'Fontname', 'Fontsize', 'PrimaryColour', 'SecondaryColour', 'OutlineColour', 'BackColour', 'Bold',
        'Italic', 'Underline', 'StrikeOut', 'ScaleX', 'ScaleY', 'Spacing', 'Angle', 'BorderStyle', 'Outline',
        'Shadow', 'Alignment', 'MarginL', 'MarginR', 'MarginV', 'Encoding'

    """

    fmt_style_dict = {'Name': 'Default', 'Fontname': 'Arial', 'Fontsize': '48', 'PrimaryColour': '&Hffffff',
                    'SecondaryColour': '&Hffffff', 'OutlineColour': '&H0', 'BackColour': '&H0', 'Bold': '0',
                    'Italic': '0', 'Underline': '0', 'StrikeOut': '0', 'ScaleX': '100', 'ScaleY': '100',
                    'Spacing': '0', 'Angle': '0', 'BorderStyle': '1', 'Outline': '1', 'Shadow': '0',
                    'Alignment': '2', 'MarginL': '10', 'MarginR': '10', 'MarginV': '10', 'Encoding': '0'}

    for k, v in filter(lambda x: 'colour' in x[0].lower() and not str(x[1]).startswith('&H'), kwargs.items()):
        kwargs[k] = f'&H{kwargs[k]}'

    fmt_style_dict.update((k, v) for k, v in kwargs.items() if k in fmt_style_dict)

    if font:
        fmt_style_dict.update(Fontname=font)
    if font_size:
        fmt_style_dict.update(Fontsize=font_size)

    fmts = f'Format: {", ".join(map(str, fmt_style_dict.keys()))}'

    styles = f'Style: {",".join(map(str, fmt_style_dict.values()))}'

    ass_str = f'[Script Info]\nScriptType: v4.00+\nPlayResX: 384\nPlayResY: 288\nScaledBorderAndShadow: yes\n\n' \
            f'[V4+ Styles]\n{fmts}\n{styles}\n\n' \
            f'[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n\n'

    if prefmt or suffmt:
        if suffmt:
            assert prefmt, 'prefmt must be used along with suffmt'
        else:
            suffmt = r'\r'
    else:
        if not color:
            color = 'HFF00'
        underline_code = r'\u1' if underline else ''

        prefmt = r'{\1c&' + f'{color.upper()}&{underline_code}' + '}'
        suffmt = r'{\r}'
    
    def secs_to_hhmmss(secs: Tuple[float, int]):
        mm, ss = divmod(secs, 60)
        hh, mm = divmod(mm, 60)
        return f'{hh:0>1.0f}:{mm:0>2.0f}:{ss:0>2.2f}'


    def dialogue(chars: str, start: float, end: float, idx_0: int, idx_1: int) -> str:
        if idx_0 == -1:
            text = chars
        else:
            text = f'{chars[:idx_0]}{prefmt}{chars[idx_0:idx_1]}{suffmt}{chars[idx_1:]}'
        return f"Dialogue: 0,{secs_to_hhmmss(start)},{secs_to_hhmmss(end)}," \
               f"Default,,0,0,0,,{text.strip() if strip else text}"

    if resolution == "word":
        resolution_key = "word-segments"
    elif resolution == "char":
        resolution_key = "char-segments"
    else:
        raise ValueError(".ass resolution should be 'word' or 'char', not ", resolution)
    
    ass_arr = []

    for segment in transcript:
        # if "12" in segment['text']:
            # import pdb; pdb.set_trace()
        if resolution_key in segment:
            res_segs = pd.DataFrame(segment[resolution_key])
            prev = segment['start']
            if "speaker" in segment:
                speaker_str = f"[{segment['speaker']}]: "
            else:
                speaker_str = ""
            for cdx, crow in res_segs.iterrows():
                if not np.isnan(crow['start']):
                    if resolution == "char":
                        idx_0 = cdx
                        idx_1 = cdx + 1
                    elif resolution == "word":
                        idx_0 = int(crow["segment-text-start"])
                        idx_1 = int(crow["segment-text-end"])
                    # fill gap
                    if crow['start'] > prev:
                        filler_ts = {
                            "chars": speaker_str + segment['text'],
                            "start": prev,
                            "end": crow['start'],
                            "idx_0": -1,
                            "idx_1": -1
                        }

                        ass_arr.append(filler_ts)
                    # highlight current word
                    f_word_ts = {
                        "chars": speaker_str + segment['text'],
                        "start": crow['start'],
                        "end": crow['end'],
                        "idx_0": idx_0 + len(speaker_str),
                        "idx_1": idx_1 + len(speaker_str)
                    }
                    ass_arr.append(f_word_ts)
                    prev = crow['end']

    ass_str += '\n'.join(map(lambda x: dialogue(**x), ass_arr))

    file.write(ass_str)

def interpolate_nans(x, method='nearest'):
    if x.notnull().sum() > 1:
        return x.interpolate(method=method).ffill().bfill()
    else:
        return x.ffill().bfill()