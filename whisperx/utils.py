import os
import zlib
from typing import Callable, TextIO, Iterator, Tuple
import pandas as pd
import numpy as np

def interpolate_nans(x, method='nearest'):
    if x.notnull().sum() > 1:
        return x.interpolate(method=method).ffill().bfill()
    else:
        return x.ffill().bfill()
    

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


from whisper.utils import SubtitlesWriter, ResultWriter, WriteTXT, WriteVTT, WriteSRT, WriteTSV, WriteJSON, format_timestamp

class WriteASS(ResultWriter):
    extension: str = "ass"

    def write_result(self, result: dict, file: TextIO):
        write_ass(result["segments"], file, resolution="word")

class WriteASSchar(ResultWriter):
    extension: str = "ass"

    def write_result(self, result: dict, file: TextIO):
        write_ass(result["segments"], file, resolution="char")

class WritePickle(ResultWriter):
    extension: str = "ass"

    def write_result(self, result: dict, file: TextIO):
        pd.DataFrame(result["segments"]).to_pickle(file)

class WriteSRTWord(ResultWriter):
    extension: str = "word.srt"
    always_include_hours: bool = True
    decimal_marker: str = ","

    def iterate_result(self, result: dict):
        for segment in result["word_segments"]:
            segment_start = self.format_timestamp(segment["start"])
            segment_end = self.format_timestamp(segment["end"])
            segment_text = segment["text"].strip().replace("-->", "->")

            if word_timings := segment.get("words", None):
                all_words = [timing["word"] for timing in word_timings]
                all_words[0] = all_words[0].strip()  # remove the leading space, if any
                last = segment_start
                for i, this_word in enumerate(word_timings):
                    start = self.format_timestamp(this_word["start"])
                    end = self.format_timestamp(this_word["end"])
                    if last != start:
                        yield last, start, segment_text

                    yield start, end, "".join(
                        [
                            f"<u>{word}</u>" if j == i else word
                            for j, word in enumerate(all_words)
                        ]
                    )
                    last = end

                if last != segment_end:
                    yield last, segment_end, segment_text
            else:
                yield segment_start, segment_end, segment_text

    def write_result(self, result: dict, file: TextIO):
        if "word_segments" not in result:
            return
        for i, (start, end, text) in enumerate(self.iterate_result(result), start=1):
            print(f"{i}\n{start} --> {end}\n{text}\n", file=file, flush=True)

    def format_timestamp(self, seconds: float):
        return format_timestamp(
            seconds=seconds,
            always_include_hours=self.always_include_hours,
            decimal_marker=self.decimal_marker,
        )

def get_writer(output_format: str, output_dir: str) -> Callable[[dict, TextIO], None]:
    writers = {
        "txt": WriteTXT,
        "vtt": WriteVTT,
        "srt": WriteSRT,
        "tsv": WriteTSV,
        "ass": WriteASS,
        "srt-word": WriteSRTWord,
        # "ass-char": WriteASSchar,
        # "pickle": WritePickle,
        # "json": WriteJSON,
    }

    writers_other = {
        "pkl": WritePickle,
        "ass-char": WriteASSchar
    }

    if output_format == "all":
        all_writers = [writer(output_dir) for writer in writers.values()]

        def write_all(result: dict, file: TextIO):
            for writer in all_writers:
                writer(result, file)

        return write_all

    if output_format in writers:
        return writers[output_format](output_dir)
    elif output_format in writers_other:
        return writers_other[output_format](output_dir)
    else:
        raise ValueError(f"Output format '{output_format}' not supported, choose from {writers.keys()} and {writers_other.keys()}")
