import os
import urllib
import pandas as pd
import numpy as np
import torch
import hashlib
from tqdm import tqdm
from typing import Optional, Callable, Union, Text
from pyannote.audio.core.io import AudioFile
from pyannote.core import Annotation, Segment, SlidingWindowFeature
from pyannote.audio.pipelines.utils import PipelineModel
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from .diarize import Segment as SegmentX
from typing import List, Tuple, Optional

VAD_SEGMENTATION_URL = "https://whisperx.s3.eu-west-2.amazonaws.com/model_weights/segmentation/0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea/pytorch_model.bin"

def load_vad_model(device, vad_onset, vad_offset, use_auth_token=None, model_fp=None):
    model_dir = torch.hub._get_torch_home()
    os.makedirs(model_dir, exist_ok = True)
    if model_fp is None:
        model_fp = os.path.join(model_dir, "whisperx-vad-segmentation.bin")
    if os.path.exists(model_fp) and not os.path.isfile(model_fp):
        raise RuntimeError(f"{model_fp} exists and is not a regular file")

    if not os.path.isfile(model_fp):
        with urllib.request.urlopen(VAD_SEGMENTATION_URL) as source, open(model_fp, "wb") as output:
            with tqdm(
                total=int(source.info().get("Content-Length")),
                ncols=80,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as loop:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break

                    output.write(buffer)
                    loop.update(len(buffer))

    model_bytes = open(model_fp, "rb").read()
    if hashlib.sha256(model_bytes).hexdigest() != VAD_SEGMENTATION_URL.split('/')[-2]:
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model."
        )

    vad_model = Model.from_pretrained(model_fp, use_auth_token=use_auth_token)
    hyperparameters = {"onset": vad_onset, 
                    "offset": vad_offset,
                    "min_duration_on": 0.1,
                    "min_duration_off": 0.1}
    vad_pipeline = VoiceActivitySegmentation(segmentation=vad_model, device=torch.device(device))
    vad_pipeline.instantiate(hyperparameters)

    return vad_pipeline

class Binarize:
    """Binarize detection scores using hysteresis thresholding, with min-cut operation
    to ensure not segments are longer than max_duration.

    Parameters
    ----------
    onset : float, optional
        Onset threshold. Defaults to 0.5.
    offset : float, optional
        Offset threshold. Defaults to `onset`.
    min_duration_on : float, optional
        Remove active regions shorter than that many seconds. Defaults to 0s.
    min_duration_off : float, optional
        Fill inactive regions shorter than that many seconds. Defaults to 0s.
    pad_onset : float, optional
        Extend active regions by moving their start time by that many seconds.
        Defaults to 0s.
    pad_offset : float, optional
        Extend active regions by moving their end time by that many seconds.
        Defaults to 0s.
    max_duration: float
        The maximum length of an active segment, divides segment at timestamp with lowest score.
    Reference
    ---------
    Gregory Gelly and Jean-Luc Gauvain. "Minimum Word Error Training of
    RNN-based Voice Activity Detection", InterSpeech 2015.

    Modified by Max Bain to include WhisperX's min-cut operation 
    https://arxiv.org/abs/2303.00747
    
    Pyannote-audio
    """

    def __init__(
        self,
        onset: float = 0.5,
        offset: Optional[float] = None,
        min_duration_on: float = 0.0,
        min_duration_off: float = 0.0,
        pad_onset: float = 0.0,
        pad_offset: float = 0.0,
        max_duration: float = float('inf')
    ):

        super().__init__()

        self.onset = onset
        self.offset = offset or onset

        self.pad_onset = pad_onset
        self.pad_offset = pad_offset

        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off

        self.max_duration = max_duration

    def __call__(self, scores: SlidingWindowFeature) -> Annotation:
        """Binarize detection scores
        Parameters
        ----------
        scores : SlidingWindowFeature
            Detection scores.
        Returns
        -------
        active : Annotation
            Binarized scores.
        """

        num_frames, num_classes = scores.data.shape
        frames = scores.sliding_window
        timestamps = [frames[i].middle for i in range(num_frames)]

        # annotation meant to store 'active' regions
        active = Annotation()
        for k, k_scores in enumerate(scores.data.T):

            label = k if scores.labels is None else scores.labels[k]

            # initial state
            start = timestamps[0]
            is_active = k_scores[0] > self.onset
            curr_scores = [k_scores[0]]
            curr_timestamps = [start]
            for t, y in zip(timestamps[1:], k_scores[1:]):
                # currently active
                if is_active: 
                    curr_duration = t - start
                    if curr_duration > self.max_duration:
                        # if curr_duration > 15:
                            # import pdb; pdb.set_trace()
                        search_after = len(curr_scores) // 2
                        # divide segment
                        min_score_div_idx = search_after + np.argmin(curr_scores[search_after:])
                        min_score_t = curr_timestamps[min_score_div_idx]
                        region = Segment(start - self.pad_onset, min_score_t + self.pad_offset)
                        active[region, k] = label
                        start = curr_timestamps[min_score_div_idx]
                        curr_scores = curr_scores[min_score_div_idx+1:]
                        curr_timestamps = curr_timestamps[min_score_div_idx+1:]
                    # switching from active to inactive
                    elif y < self.offset:
                        region = Segment(start - self.pad_onset, t + self.pad_offset)
                        active[region, k] = label
                        start = t
                        is_active = False
                        curr_scores = []
                        curr_timestamps = []
                # currently inactive
                else:
                    # switching from inactive to active
                    if y > self.onset:
                        start = t
                        is_active = True
                curr_scores.append(y)
                curr_timestamps.append(t)

            # if active at the end, add final region
            if is_active:
                region = Segment(start - self.pad_onset, t + self.pad_offset)
                active[region, k] = label

        # because of padding, some active regions might be overlapping: merge them.
        # also: fill same speaker gaps shorter than min_duration_off
        if self.pad_offset > 0.0 or self.pad_onset > 0.0 or self.min_duration_off > 0.0:
            if self.max_duration < float("inf"):
                raise NotImplementedError(f"This would break current max_duration param")
            active = active.support(collar=self.min_duration_off)

        # remove tracks shorter than min_duration_on
        if self.min_duration_on > 0:
            for segment, track in list(active.itertracks()):
                if segment.duration < self.min_duration_on:
                    del active[segment, track]

        return active


class VoiceActivitySegmentation(VoiceActivityDetection):
    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation",
        fscore: bool = False,
        use_auth_token: Union[Text, None] = None,
        **inference_kwargs,
    ):

        super().__init__(segmentation=segmentation, fscore=fscore, use_auth_token=use_auth_token, **inference_kwargs)

    def apply(self, file: AudioFile, hook: Optional[Callable] = None) -> Annotation:
        """Apply voice activity detection

        Parameters
        ----------
        file : AudioFile
            Processed file.
        hook : callable, optional
            Hook called after each major step of the pipeline with the following
            signature: hook("step_name", step_artefact, file=file)

        Returns
        -------
        speech : Annotation
            Speech regions.
        """

        # setup hook (e.g. for debugging purposes)
        hook = self.setup_hook(file, hook=hook)

        # apply segmentation model (only if needed)
        # output shape is (num_chunks, num_frames, 1)
        if self.training:
            if self.CACHED_SEGMENTATION in file:
                segmentations = file[self.CACHED_SEGMENTATION]
            else:
                segmentations = self._segmentation(file)
                file[self.CACHED_SEGMENTATION] = segmentations
        else:
            segmentations: SlidingWindowFeature = self._segmentation(file)

        return segmentations


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

def merge_chunks(segments, chunk_size):
    """
    Merge operation described in paper
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
        segments_list.append(SegmentX(speech_turn.start, speech_turn.end, "UNKNOWN"))

    if len(segments_list) == 0:
        print("No active speech found in audio")
        return []
    # assert segments_list, "segments_list is empty."
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
