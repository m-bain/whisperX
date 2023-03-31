import numpy as np
import pandas as pd
from pyannote.audio import Pipeline

class DiarizationPipeline:
    def __init__(
        self,
        model_name="pyannote/speaker-diarization@2.1",
        use_auth_token=None,
    ):
        self.model = Pipeline.from_pretrained(model_name, use_auth_token=use_auth_token)

    def __call__(self, audio, min_speakers=None, max_speakers=None):
        segments = self.model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
        diarize_df = pd.DataFrame(segments.itertracks(yield_label=True))
        diarize_df['start'] = diarize_df[0].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df[0].apply(lambda x: x.end)
        return diarize_df

def assign_word_speakers(diarize_df, result_segments, fill_nearest=False):
    for seg in result_segments:
        wdf = seg['word-segments']
        if len(wdf['start'].dropna()) == 0:
            wdf['start'] = seg['start']
            wdf['end'] = seg['end']
        speakers = []
        for wdx, wrow in wdf.iterrows():
            if not np.isnan(wrow['start']):
                diarize_df['intersection'] = np.minimum(diarize_df['end'], wrow['end']) - np.maximum(diarize_df['start'], wrow['start'])
                diarize_df['union'] = np.maximum(diarize_df['end'], wrow['end']) - np.minimum(diarize_df['start'], wrow['start'])
                # remove no hit
                if not fill_nearest:
                    dia_tmp = diarize_df[diarize_df['intersection'] > 0]
                else:
                    dia_tmp = diarize_df
                if len(dia_tmp) == 0:
                    speaker = None
                else:
                    speaker = dia_tmp.sort_values("intersection", ascending=False).iloc[0][2]
            else:
                speaker = None
            speakers.append(speaker)
        seg['word-segments']['speaker'] = speakers

        speaker_count = pd.Series(speakers).value_counts()
        if len(speaker_count) == 0:
            seg["speaker"]= "UNKNOWN"
        else:
            seg["speaker"] = speaker_count.index[0]

    # create word level segments for .srt
    word_seg = []
    for seg in result_segments:
        wseg = pd.DataFrame(seg["word-segments"])
        for wdx, wrow in wseg.iterrows():
            if wrow["start"] is not None:
                speaker = wrow['speaker']
                if speaker is None or speaker == np.nan:
                    speaker = "UNKNOWN"
                word_seg.append(
                    {
                        "start": wrow["start"],
                        "end": wrow["end"],
                        "text": f"[{speaker}]: " + seg["text"][int(wrow["segment-text-start"]):int(wrow["segment-text-end"])]
                    }
                )

    # TODO: create segments but split words on new speaker

    return result_segments, word_seg

class Segment:
    def __init__(self, start, end, speaker=None):
        self.start = start
        self.end = end
        self.speaker = speaker
