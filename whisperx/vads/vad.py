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

    # Unused function
    @staticmethod
    def merge_vad(vad_arr, pad_onset=0.0, pad_offset=0.0, min_duration_off=0.0, min_duration_on=0.0):
        active = Annotation()
        for k, vad_t in enumerate(vad_arr):
            region = Segment(vad_t[0] - pad_onset, vad_t[1] + pad_offset)
            active[region, k] = 1

        if pad_offset > 0.0 or pad_onset > 0.0 or min_duration_off > 0.0:
            active = active.support(collar=min_duration_off)

        # remove tracks shorter than min_duration_on
        if min_duration_on > 0:
            for segment, track in list(active.itertracks()):
                if segment.duration < min_duration_on:
                    del active[segment, track]

        active = active.for_json()
        active_segs = pd.DataFrame([x['segment'] for x in active['content']])
        return active_segs
