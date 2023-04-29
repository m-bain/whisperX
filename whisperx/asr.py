import os
import warnings
from typing import List, Union

import ctranslate2
import faster_whisper
import numpy as np
import torch
from transformers import Pipeline
from transformers.pipelines.pt_utils import PipelineIterator

from .audio import N_SAMPLES, SAMPLE_RATE, load_audio, log_mel_spectrogram
from .vad import load_vad_model, merge_chunks


def load_model(whisper_arch, device, compute_type="float16", asr_options=None, language=None,
               vad_options=None, model=None):
    '''Load a Whisper model for inference.
    Args:
        whisper_arch: str - The name of the Whisper model to load.
        device: str - The device to load the model on.
        compute_type: str - The compute type to use for the model.
        options: dict - A dictionary of options to use for the model.
        language: str - The language of the model. (use English for now)
    Returns:
        A Whisper pipeline.
    '''    

    if whisper_arch.endswith(".en"):
        language = "en"

    model = WhisperModel(whisper_arch, device=device, compute_type=compute_type)
    if language is not None:
        tokenizer = faster_whisper.tokenizer.Tokenizer(model.hf_tokenizer, model.model.is_multilingual, task="transcribe", language=language)
    else:
        print("No language specified, language will be first be detected for each audio file (increases inference time).")
        tokenizer = None

    default_asr_options =  {
        "beam_size": 5,
        "best_of": 5,
        "patience": 1,
        "length_penalty": 1,
        "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False,
        "initial_prompt": None,
        "prefix": None,
        "suppress_blank": True,
        "suppress_tokens": [-1],
        "without_timestamps": True,
        "max_initial_timestamp": 0.0,
        "word_timestamps": False,
        "prepend_punctuations": "\"'“¿([{-",
        "append_punctuations": "\"'.。,，!！?？:：”)]}、"
    }

    if asr_options is not None:
        default_asr_options.update(asr_options)
    default_asr_options = faster_whisper.transcribe.TranscriptionOptions(**default_asr_options)

    default_vad_options = {
        "vad_onset": 0.500,
        "vad_offset": 0.363
    }

    if vad_options is not None:
        default_vad_options.update(vad_options)

    vad_model = load_vad_model(torch.device(device), use_auth_token=None, **default_vad_options)

    return FasterWhisperPipeline(model, vad_model, default_asr_options, tokenizer)



class WhisperModel(faster_whisper.WhisperModel):
    '''
    FasterWhisperModel provides batched inference for faster-whisper.
    Currently only works in non-timestamp mode.
    '''

    def generate_segment_batched(self, features: np.ndarray, tokenizer: faster_whisper.tokenizer.Tokenizer, options: faster_whisper.transcribe.TranscriptionOptions, encoder_output = None):
        batch_size = features.shape[0]
        all_tokens = []
        prompt_reset_since = 0
        if options.initial_prompt is not None:
            initial_prompt = " " + options.initial_prompt.strip()
            initial_prompt_tokens = tokenizer.encode(initial_prompt)
            all_tokens.extend(initial_prompt_tokens)
        previous_tokens = all_tokens[prompt_reset_since:]
        prompt = self.get_prompt(
            tokenizer,
            previous_tokens,
            without_timestamps=options.without_timestamps,
            prefix=options.prefix,
        )

        encoder_output = self.encode(features)

        max_initial_timestamp_index = int(
            round(options.max_initial_timestamp / self.time_precision)
        )

        result = self.model.generate(
                encoder_output,
                [prompt] * batch_size,
                # length_penalty=options.length_penalty,
                # max_length=self.max_length,
                # return_scores=True,
                # return_no_speech_prob=True,
                # suppress_blank=options.suppress_blank,
                # suppress_tokens=options.suppress_tokens,
                # max_initial_timestamp_index=max_initial_timestamp_index,
            )
    
        tokens_batch = [x.sequences_ids[0] for x in result]

        def decode_batch(tokens: List[List[int]]) -> str:
            res = []
            for tk in tokens:
                res.append([token for token in tk if token < tokenizer.eot])
            # text_tokens = [token for token in tokens if token < self.eot]
            return tokenizer.tokenizer.decode_batch(res)

        text = decode_batch(tokens_batch)

        return text
    
    def encode(self, features: np.ndarray) -> ctranslate2.StorageView:
        # When the model is running on multiple GPUs, the encoder output should be moved
        # to the CPU since we don't know which GPU will handle the next job.
        to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1
        # unsqueeze if batch size = 1
        if len(features.shape) == 2:
            features = np.expand_dims(features, 0)
        features = faster_whisper.transcribe.get_ctranslate2_storage(features)
        
        return self.model.encode(features, to_cpu=to_cpu)
    
class FasterWhisperPipeline(Pipeline):
    def __init__(
            self,
            model,
            vad,
            options,
            tokenizer=None,
            device: Union[int, str, "torch.device"] = -1,
            framework = "pt",
            **kwargs
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.options = options
        self._batch_size = kwargs.pop("batch_size", None)
        self._num_workers = 1
        self._preprocess_params, self._forward_params, self._postprocess_params = self._sanitize_parameters(**kwargs)
        self.call_count = 0
        self.framework = framework
        if self.framework == "pt":
            if isinstance(device, torch.device):
                self.device = device
            elif isinstance(device, str):
                self.device = torch.device(device)
            elif device < 0:
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(f"cuda:{device}")
        else:
            self.device = device
        
        super(Pipeline, self).__init__()
        self.vad_model = vad

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "tokenizer" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, audio):
        audio = audio['inputs']
        features = log_mel_spectrogram(audio, padding=N_SAMPLES - audio.shape[0])
        return {'inputs': features}

    def _forward(self, model_inputs):
        outputs = self.model.generate_segment_batched(model_inputs['inputs'], self.tokenizer, self.options)
        return {'text': outputs}
    
    def postprocess(self, model_outputs):
        return model_outputs

    def get_iterator(
        self, inputs, num_workers: int, batch_size: int, preprocess_params, forward_params, postprocess_params
    ):
        dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # TODO hack by collating feature_extractor and image_processor

        def stack(items):
            return {'inputs': torch.stack([x['inputs'] for x in items])}
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=stack)
        model_iterator = PipelineIterator(dataloader, self.forward, forward_params, loader_batch_size=batch_size)
        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
        return final_iterator

    def transcribe(
        self, audio: Union[str, np.ndarray], batch_size=None, num_workers=0
    ):
        if isinstance(audio, str):
            audio = load_audio(audio)
        
        def data(audio, segments):
            for seg in segments:
                f1 = int(seg['start'] * SAMPLE_RATE)
                f2 = int(seg['end'] * SAMPLE_RATE)
                # print(f2-f1)
                yield {'inputs': audio[f1:f2]}

        vad_segments = self.vad_model({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SAMPLE_RATE})
        vad_segments = merge_chunks(vad_segments, 30)

        del_tokenizer = False
        if self.tokenizer is None:
            language = self.detect_language(audio)
            self.tokenizer = faster_whisper.tokenizer.Tokenizer(self.model.hf_tokenizer, self.model.model.is_multilingual, task="transcribe", language=language)
            del_tokenizer = True
        else:
            language = self.tokenizer.language_code

        segments = []
        batch_size = batch_size or self._batch_size
        for idx, out in enumerate(self.__call__(data(audio, vad_segments), batch_size=batch_size, num_workers=num_workers)):
            text = out['text']
            if batch_size in [0, 1, None]:
                text = text[0]
            segments.append(
                {
                    "text": out['text'],
                    "start": round(vad_segments[idx]['start'], 3),
                    "end": round(vad_segments[idx]['end'], 3)
                }
            )
        
        if del_tokenizer:
            self.tokenizer = None

        return {"segments": segments, "language": language}


    def detect_language(self, audio: np.ndarray):
        segment = log_mel_spectrogram(audio[: N_SAMPLES], padding=0)
        encoder_output = self.model.encode(segment)
        results = self.model.model.detect_language(encoder_output)
        language_token, language_probability = results[0][0]
        language = language_token[2:-2]
        print(f"Detected language: {language} ({language_probability:.2f}) in first 30s of audio...")
        return language

if __name__ == "__main__":
    main_type = "simple"
    import time

    import jiwer
    from tqdm import tqdm
    from whisper.normalizers import EnglishTextNormalizer

    from benchmark.tedlium import parse_tedlium_annos

    if main_type == "complex":
        from faster_whisper.tokenizer import Tokenizer
        from faster_whisper.transcribe import TranscriptionOptions
        from faster_whisper.vad import (SpeechTimestampsMap,
                                        get_speech_timestamps)

        from whisperx.vad import load_vad_model, merge_chunks

        from .audio import SAMPLE_RATE, load_audio, log_mel_spectrogram
        faster_t_options = TranscriptionOptions(
        beam_size=5,
        best_of=5,
        patience=1,
        length_penalty=1,
        temperatures=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        condition_on_previous_text=False,
        initial_prompt=None,
        prefix=None,
        suppress_blank=True,
        suppress_tokens=[-1],
        without_timestamps=True,
        max_initial_timestamp=0.0,
        word_timestamps=False,
        prepend_punctuations="\"'“¿([{-",
        append_punctuations="\"'.。,，!！?？:：”)]}、"
    )
        whisper_arch = "large-v2"
        device = "cuda"
        batch_size = 16
        model = WhisperModel(whisper_arch, device="cuda", compute_type="float16",)
        tokenizer = Tokenizer(model.hf_tokenizer, model.model.is_multilingual, task="transcribe", language="en")
        model = FasterWhisperPipeline(model, tokenizer, faster_t_options, device=-1)
        fn = "DanielKahneman_2010.wav"
        wav_dir = f"/tmp/test/wav/"
        vad_model = load_vad_model("cuda", 0.6, 0.3)
        audio = load_audio(os.path.join(wav_dir, fn))
        vad_segments = vad_model({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SAMPLE_RATE})
        vad_segments = merge_chunks(vad_segments, 30)

        def data(audio, segments):
            for seg in segments:
                f1 = int(seg['start'] * SAMPLE_RATE)
                f2 = int(seg['end'] * SAMPLE_RATE)
                # print(f2-f1)
                yield {'inputs': audio[f1:f2]}
        vad_method="pyannote"

        wav_dir = f"/tmp/test/wav/"
        wer_li = []
        time_li = []
        for fn in os.listdir(wav_dir):
            if fn == "RobertGupta_2010U.wav":
                continue
            base_fn = fn.split('.')[0]
            audio_fp = os.path.join(wav_dir, fn)

            audio = load_audio(audio_fp)
            t1 = time.time()
            if vad_method == "pyannote":
                vad_segments = vad_model({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SAMPLE_RATE})
                vad_segments = merge_chunks(vad_segments, 30)
            elif vad_method == "silero":
                vad_segments = get_speech_timestamps(audio, threshold=0.5, max_speech_duration_s=30)
                vad_segments = [{"start": x["start"] / SAMPLE_RATE, "end": x["end"] / SAMPLE_RATE} for x in vad_segments]
                new_segs = []
                curr_start = vad_segments[0]['start']
                curr_end = vad_segments[0]['end']
                for seg in vad_segments[1:]:
                    if seg['end'] - curr_start > 30:
                        new_segs.append({"start": curr_start, "end": curr_end})
                        curr_start = seg['start']
                        curr_end = seg['end']
                    else:
                        curr_end = seg['end']
                new_segs.append({"start": curr_start, "end": curr_end})
                vad_segments = new_segs
            text = []
            # for idx, out in tqdm(enumerate(model(data(audio_fp, vad_segments), batch_size=batch_size)), total=len(vad_segments)):
            for idx, out in enumerate(model(data(audio, vad_segments), batch_size=batch_size)):
                text.append(out['text'])
            t2 = time.time()
            if batch_size == 1:
                text = [x[0] for x in text]
            text = " ".join(text)

            normalizer = EnglishTextNormalizer()
            text = normalizer(text)
            gt_corpus = normalizer(parse_tedlium_annos(base_fn, "/tmp/test/"))

            wer_result = jiwer.wer(gt_corpus, text)
            print("WER: %.2f \t time: %.2f \t [%s]" % (wer_result * 100, t2-t1, fn))

            wer_li.append(wer_result)
            time_li.append(t2-t1)
        print("# Avg Mean...")
        print("WER: %.2f" % (sum(wer_li) * 100/len(wer_li)))
        print("Time: %.2f" % (sum(time_li)/len(time_li)))
    elif main_type == "simple":
        model = load_model(
            "large-v2",
            device="cuda",
            language="en",
        )

        wav_dir = f"/tmp/test/wav/"
        wer_li = []
        time_li = []
        for fn in os.listdir(wav_dir):
            if fn == "RobertGupta_2010U.wav":
                continue
            # fn = "DanielKahneman_2010.wav"
            base_fn = fn.split('.')[0]
            audio_fp = os.path.join(wav_dir, fn)

            audio = load_audio(audio_fp)
            t1 = time.time()
            out = model.transcribe(audio_fp, batch_size=8)["segments"]
            t2 = time.time()

            text = " ".join([x['text'] for x in out])
            normalizer = EnglishTextNormalizer()
            text = normalizer(text)
            gt_corpus = normalizer(parse_tedlium_annos(base_fn, "/tmp/test/"))

            wer_result = jiwer.wer(gt_corpus, text)
            print("WER: %.2f \t time: %.2f \t [%s]" % (wer_result * 100, t2-t1, fn))

            wer_li.append(wer_result)
            time_li.append(t2-t1)
        print("# Avg Mean...")
        print("WER: %.2f" % (sum(wer_li) * 100/len(wer_li)))
        print("Time: %.2f" % (sum(time_li)/len(time_li)))
