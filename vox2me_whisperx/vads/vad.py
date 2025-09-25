from typing import Optional

import pandas as pd
from pyannote.core import Annotation, Segment


class Vad:
    def __init__(self, vad_onset):
        if not (0 < vad_onset < 1):
            raise ValueError(
                "vad_onset is a decimal value between 0 and 1."
            )

    @staticmethod
    def preprocess_audio(audio):
        pass

    # keep merge_chunks as static so it can be also used by manually assigned vad_model (see 'load_model')
    @staticmethod
    def merge_chunks(segments,
                     chunk_size,
                     onset: float,
                     offset: Optional[float]):
        """
         Merge operation described in paper
         """
        curr_end = 0
        merged_segments = []
        seg_idxs: list[tuple]= []
        speaker_idxs: list[Optional[str]] = []

        curr_start = segments[0].start
        for seg in segments:
            if seg.end - curr_start > chunk_size and curr_end - curr_start > 0:
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

