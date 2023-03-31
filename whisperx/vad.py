import numpy as np
from pyannote.core import Annotation, SlidingWindowFeature
from typing import List, Optional, Union, BinaryIO, NamedTuple
import torch
from abc import ABC, abstractmethod
from faster_whisper.transcribe import Word


class FasterWhisperSegment(NamedTuple):
    """Merged all potential fields in WhisperX and Faster-Whisper's Segment"""
    start: float
    end: float
    text: str
    speaker: Optional[str]
    words: Optional[List[Word]]


class VADPipeline(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def get_segments(self, audio: Union[str, BinaryIO, np.ndarray], chunk_size: int = 30):
        pass
    
    @staticmethod
    def merge_chunks(segments_list, chunk_size = 30):
        """
        Merge VAD segments into larger segments of approximately size ~CHUNK_LENGTH.
        TODO: Make sure VAD segment isn't too long, otherwise it will cause OOM when input to alignment model
        TODO: Or sliding window alignment model over long segment.
        """
        curr_end = 0
        merged_segments = []
        seg_idxs = []
        speaker_idxs = []
        
        assert chunk_size > 0
        
        assert segments_list, "segments_list is empty."
        # Make sure the starting point is the start of the segment.
        curr_start = segments_list[0].start
        
        for seg in segments_list:
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


class PyannoteVADPipeline(VADPipeline):
    def __init__(self, hf_token, device: str = 'cpu'):
        super().__init__()
        from pyannote.audio import Inference, Model
        self.vad_pipeline = Inference(
                Model.from_pretrained("pyannote/segmentation",
                                      use_auth_token = hf_token),
                pre_aggregation_hook = lambda segmentation: segmentation,
                use_auth_token = hf_token,
                device = torch.device(device),
        )
        pass
    
    def get_segments(self, audio: Union[str, BinaryIO, np.ndarray], chunk_size: int = 30):
        """use pyannote to get segments of speech"""
        segments = self.vad_pipeline(audio)
        return self.merge_chunks(segments, chunk_size)


class SileroVADPipeline(VADPipeline):
    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.vad_pipeline, vad_utils = torch.hub.load(repo_or_dir = 'snakers4/silero-vad',
                                                      model = 'silero_vad',
                                                      force_reload = True,
                                                      onnx = False,
                                                      trust_repo = True, )
        (self.get_speech_timestamps, _, self.read_audio, _, _) = vad_utils
        self.device = device
    
    def get_segments(self, audio: Union[str, BinaryIO, np.ndarray],
                     chunk_size: int = 30):
        """use silero to get segments of speech"""
        sample_rate = 16000  # default sample rate per silero-vad
        # If user provides a numpy array, we need to know the sample rate as cannot infer from array
        if isinstance(audio, np.ndarray):
            audio_array = audio
        else:
            audio_array = self.read_audio(audio, sampling_rate = sample_rate)
        
        # https://github.com/snakers4/silero-vad/wiki/Examples-and-Dependencies
        timestamps = self.get_speech_timestamps(audio_array,
                                                model = self.vad_pipeline,
                                                sampling_rate = sample_rate)
        # sample output: [{'end': 664992, 'start': 181344}, {'end': 1373088, 'start': 672864}]
        # Segment defined here is in seconds, following the pyannote convention
        segments = [FasterWhisperSegment(i['start'] / sample_rate, i['end'] / sample_rate, "", "", []) for i in timestamps]
        return self.merge_chunks(segments, chunk_size)


class Binarize:
    """Binarize detection scores using hysteresis thresholding
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
                        region = FasterWhisperSegment(start - self.pad_onset, min_score_t + self.pad_offset)
                        active[region, k] = label
                        start = curr_timestamps[min_score_div_idx]
                        curr_scores = curr_scores[min_score_div_idx + 1:]
                        curr_timestamps = curr_timestamps[min_score_div_idx + 1:]
                    # switching from active to inactive
                    elif y < self.offset:
                        region = FasterWhisperSegment(start - self.pad_onset, t + self.pad_offset)
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
                region = FasterWhisperSegment(start - self.pad_onset, t + self.pad_offset)
                active[region, k] = label
        
        # because of padding, some active regions might be overlapping: merge them.
        # also: fill same speaker gaps shorter than min_duration_off
        if self.pad_offset > 0.0 or self.pad_onset > 0.0 or self.min_duration_off > 0.0:
            if self.max_duration < float("inf"):
                raise NotImplementedError(f"This would break current max_duration param")
            active = active.support(collar = self.min_duration_off)
        
        # remove tracks shorter than min_duration_on
        if self.min_duration_on > 0:
            for segment, track in list(active.itertracks()):
                if segment.duration < self.min_duration_on:
                    del active[segment, track]
        
        return active


def merge_vad(vad_arr, pad_onset = 0.0, pad_offset = 0.0, min_duration_off = 0.0, min_duration_on = 0.0):
    
    active = Annotation()
    for k, vad_t in enumerate(vad_arr):
        region = FasterWhisperSegment(vad_t[0] - pad_onset, vad_t[1] + pad_offset)
        active[region, k] = 1
    
    if pad_offset > 0.0 or pad_onset > 0.0 or min_duration_off > 0.0:
        active = active.support(collar = min_duration_off)
    
    # remove tracks shorter than min_duration_on
    if min_duration_on > 0:
        for segment, track in list(active.itertracks()):
            if segment.duration < min_duration_on:
                del active[segment, track]
    
    active = active.for_json()
    active_segs = pd.DataFrame([x['segment'] for x in active['content']])
    return active_segs


if __name__ == "__main__":
    # from pyannote.audio import Inference
    # hook = lambda segmentation: segmentation
    # inference = Inference("pyannote/segmentation", pre_aggregation_hook=hook)
    # audio = "/tmp/11962.wav" 
    # scores = inference(audio)
    # binarize = Binarize(max_duration=15)
    # anno = binarize(scores)
    # res = []
    # for ann in anno.get_timeline():
    #     res.append((ann.start, ann.end))
    
    # res = pd.DataFrame(res)
    # res[2] = res[1] - res[0]
    import pandas as pd
    
    input_fp = "tt298650_sync.wav"
    df = pd.read_csv(f"/work/maxbain/tmp/{input_fp}.sad", sep = " ", header = None)
    print(len(df))
    N = 0.15
    g = df[0].sub(df[1].shift())
    input_base = input_fp.split('.')[0]
    df = df.groupby(g.gt(N).cumsum()).agg({0: 'min', 1: 'max'})
    df.to_csv(f"/work/maxbain/tmp/{input_base}.lab", header = None, index = False, sep = " ")
    print(df)
    import pdb ;
    
    pdb.set_trace()
