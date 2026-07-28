import argparse
import importlib.metadata
import platform
import os
import json

from utils import (optional_float, optional_int, str2bool, get_writer)

def cli():
    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("json", nargs="+", type=str, help="transcripts (whisperx-produced .JSON's) to convert into subtitles")

    parser.add_argument("--output_dir", "-o", type=str, default=".", help="directory to save the outputs")
    parser.add_argument("--output_format", "-f", type=str, default="all", choices=["all", "srt", "vtt", "txt", "tsv", "json", "aud"], help="format of the output file; if not specified, all available formats will be produced")

    parser.add_argument("--max_line_width", type=optional_int, default=None, help="the maximum number of characters in a line before breaking the line")
    parser.add_argument("--max_line_count", type=optional_int, default=None, help="the maximum number of lines in a segment")
    parser.add_argument("--max_segment_duration", type=optional_float, default=None, help="the maximum duration of a segment")
    parser.add_argument("--max_short_pause_duration", type=optional_float, default=None, help="if there is a pause (silence) of this duration, start a new segment")
    parser.add_argument("--highlight_words", type=str2bool, default=False, help="underline each word as it is spoken in srt and vtt")
    # fmt: on

    args = parser.parse_args().__dict__

    output_dir: str = args.pop("output_dir")
    output_format: str = args.pop("output_format")

    os.makedirs(output_dir, exist_ok=True)

    writer = get_writer(output_format, output_dir)
    word_options = ["highlight_words", "max_line_count", "max_line_width", "max_segment_duration", "max_short_pause_duration"]
    if args["max_line_count"] and not args["max_line_width"]:
        warnings.warn("--max_line_count has no effect without --max_line_width")
    writer_args = {arg: args.pop(arg) for arg in word_options}

    for json_path in args.pop("json"):
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        writer(data, json_path, writer_args)


if __name__ == "__main__":
    cli()
