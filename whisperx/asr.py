import os
import warnings
from typing import List, Optional, Union
from dataclasses import replace

import ctranslate2
import faster_whisper
import numpy as np
import torch
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import TranscriptionOptions, get_ctranslate2_storage
from transformers import Pipeline
from transformers.pipelines.pt_utils import PipelineIterator

from whisperx.audio import N_SAMPLES, SAMPLE_RATE, load_audio, log_mel_spectrogram
from whisperx.schema import SingleSegment, TranscriptionResult, ProgressCallback
from whisperx.vads import Vad, Silero, Pyannote
from whisperx.log_utils import get_logger

logger = get_logger(__name__)


def find_numeral_symbol_tokens(tokenizer):
    numeral_symbol_tokens = []
    for i in range(tokenizer.eot):
        token = tokenizer.decode([i]).removeprefix(" ")
        has_numeral_symbol = any(c in "0123456789%$£" for c in token)
        if has_numeral_symbol:
            numeral_symbol_tokens.append(i)
    return numeral_symbol_tokens

class WhisperModel(faster_whisper.WhisperModel):
    '''
    FasterWhisperModel provides batched inference for faster-whisper.
    Currently only works in non-timestamp mode and fixed prompt for all samples in batch.
    '''

    def generate_segment_batched(
        self,
        features: np.ndarray,
        tokenizer: Tokenizer,
        options: TranscriptionOptions,
        encoder_output=None,
        previous_batch_context_tokens: List[List[int]] = None,
    ):
        batch_size = features.shape[0]
        if previous_batch_context_tokens is None:
            previous_batch_context_tokens = [[] for _ in range(batch_size)]

        initial_prompt_tokens = []
        if options.initial_prompt is not None:
            initial_prompt = " " + options.initial_prompt.strip()
            initial_prompt_tokens = tokenizer.encode(initial_prompt)

        batch_tokens = []
        for i in range(batch_size):
            all_tokens = list(initial_prompt_tokens)
            if i < len(previous_batch_context_tokens):
                ctx = previous_batch_context_tokens[i]
                if ctx:
                    available = 224 - len(all_tokens)
                    if available > 0:
                        ctx_trimmed = ctx[-available:] if len(ctx) > available else ctx
                        all_tokens.extend(ctx_trimmed)
            batch_tokens.append(all_tokens)

        max_batch_tokens = max([len(t) for t in batch_tokens] + [0])

        prompts = [
            self.get_prompt(
                tokenizer,
                [tokenizer.eot] * (max_batch_tokens - len(t)) + t,
                without_timestamps=options.without_timestamps,
                prefix=options.prefix,
                hotwords=options.hotwords
            ) for t in batch_tokens
        ]

        encoder_output = self.encode(features)

        result = self.model.generate(
                encoder_output,
                prompts,
                beam_size=options.beam_size,
                patience=options.patience,
                length_penalty=options.length_penalty,
                max_length=self.max_length,
                suppress_blank=options.suppress_blank,
                suppress_tokens=options.suppress_tokens,
                no_repeat_ngram_size=options.no_repeat_ngram_size,
                repetition_penalty=options.repetition_penalty,
                return_scores=True,
            )
        
        tokens_batch = [x.sequences_ids[0] for x in result]

        avg_logprobs = []
        for res in result:
            seq_len = len(res.sequences_ids[0])
            cum_logprob = res.scores[0] * (seq_len ** options.length_penalty)
            avg_logprobs.append(cum_logprob / (seq_len + 1))

        def decode_batch(tokens: List[List[int]]) -> List[str]:
            res = []
            for tk in tokens:
                res.append([token for token in tk if token < tokenizer.eot])
            return tokenizer.tokenizer.decode_batch(res)

        text = decode_batch(tokens_batch)

        filtered_tokens = [[t for t in tk if t < tokenizer.eot] for tk in tokens_batch]
        context_tokens_len = [len(t) for t in batch_tokens]

        return {
            'text': text,
            'avg_logprob': avg_logprobs,
            'tokens': filtered_tokens,
            'context_tokens_len': context_tokens_len,
        }

    def encode(self, features: np.ndarray) -> ctranslate2.StorageView:
        # When the model is running on multiple GPUs, the encoder output should be moved
        # to the CPU since we don't know which GPU will handle the next job.
        to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1
        # unsqueeze if batch size = 1
        if len(features.shape) == 2:
            features = np.expand_dims(features, 0)
        features = get_ctranslate2_storage(features)

        return self.model.encode(features, to_cpu=to_cpu)

class FasterWhisperPipeline(Pipeline):
    """
    Huggingface Pipeline wrapper for FasterWhisperModel.
    """
    # TODO:
    # - add support for timestamp mode
    # - add support for custom inference kwargs

    def __init__(
        self,
        model: WhisperModel,
        vad,
        vad_params: dict,
        options: TranscriptionOptions,
        tokenizer: Optional[Tokenizer] = None,
        device: Union[int, str, "torch.device"] = -1,
        framework="pt",
        language: Optional[str] = None,
        suppress_numerals: bool = False,
        **kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.options = options
        self.preset_language = language
        self.suppress_numerals = suppress_numerals
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
        self._vad_params = vad_params
        self.previous_batch_context_tokens = []
        self.last_segment_tokens_per_stream = []
        self.stream_segment_indices = []
        self.segment_output_tokens = {}  # Track output tokens per stream:segment for verification
        self.batch_counter = 0  # Track batch boundaries
        self._use_interleaved_context = False  # Internal flag: set True by transcribe() when interleaved context is active

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "tokenizer" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, audio):
        audio = audio['inputs']
        model_n_mels = self.model.feat_kwargs.get("feature_size")
        features = log_mel_spectrogram(
            audio,
            n_mels=model_n_mels if model_n_mels is not None else 80,
            padding=N_SAMPLES - audio.shape[0],
        )
        return {'inputs': features}

    def _forward(self, model_inputs):
        current_batch_size = model_inputs['inputs'].shape[0]
        valid_contexts = self.previous_batch_context_tokens[:current_batch_size]

        self.batch_counter += 1

        outputs = self.model.generate_segment_batched(
            model_inputs['inputs'],
            self.tokenizer,
            self.options,
            previous_batch_context_tokens=valid_contexts,
        )

        # Rolling context update: accumulate output tokens per stream (only when interleaved context is active)
        if self._use_interleaved_context:
            initial_prompt_length = 0
            if self.options.initial_prompt is not None:
                initial_prompt = " " + self.options.initial_prompt.strip()
                initial_prompt_length = len(self.tokenizer.encode(initial_prompt))

            max_context_window = max(0, 224 - initial_prompt_length)

            for i in range(current_batch_size):
                if i < len(self.previous_batch_context_tokens):
                    tokens = outputs['tokens'][i]
                    self.last_segment_tokens_per_stream[i] = tokens
                    self.previous_batch_context_tokens[i].extend(tokens)
                    self.previous_batch_context_tokens[i] = self.previous_batch_context_tokens[i][-max_context_window:]
                    self.stream_segment_indices[i] += 1

        return outputs

    def postprocess(self, model_outputs):
        return model_outputs

    def get_iterator(
        self,
        inputs,
        num_workers: int,
        batch_size: int,
        preprocess_params: dict,
        forward_params: dict,
        postprocess_params: dict,
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
        self,
        audio: Union[str, np.ndarray],
        batch_size: Optional[int] = None,
        num_workers=0,
        language: Optional[str] = None,
        task: Optional[str] = None,
        chunk_size=30,
        print_progress=False,
        combined_progress=False,
        verbose=False,
        progress_callback: ProgressCallback = None,
        interleaved_context: bool = False,
    ) -> TranscriptionResult:

        if isinstance(audio, str):
            audio = load_audio(audio)
        
        batch_size = batch_size or self._batch_size
        # Initialize context for each stream. 
        # We have 'batch_size' concurrent streams.
        if batch_size is None or batch_size < 1:
            batch_size = 1

        # Gate all interleaved-context machinery on the interleaved_context flag
        self._use_interleaved_context = interleaved_context and batch_size > 1

        self.previous_batch_context_tokens = [[] for _ in range(batch_size)]
        self.last_segment_tokens_per_stream = [[] for _ in range(batch_size)]
        self.stream_segment_indices = [0 for _ in range(batch_size)]

        def data(audio, segments):
            for seg in segments:
                f1 = int(seg['start'] * SAMPLE_RATE)
                f2 = int(seg['end'] * SAMPLE_RATE)
                # print(f2-f1)
                yield {'inputs': audio[f1:f2]}

        # Pre-process audio and merge chunks as defined by the respective VAD child class 
        # In case vad_model is manually assigned (see 'load_model') follow the functionality of pyannote toolkit
        if issubclass(type(self.vad_model), Vad):
            waveform = self.vad_model.preprocess_audio(audio)
            merge_chunks =  self.vad_model.merge_chunks
        else:
            waveform = Pyannote.preprocess_audio(audio)
            merge_chunks = Pyannote.merge_chunks

        vad_segments = self.vad_model({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        vad_segments = merge_chunks(
            vad_segments,
            chunk_size,
            onset=self._vad_params["vad_onset"],
            offset=self._vad_params["vad_offset"],
        )
        if self.tokenizer is None:
            language = language or self.detect_language(audio)
            task = task or "transcribe"
            self.tokenizer = Tokenizer(
                self.model.hf_tokenizer,
                self.model.model.is_multilingual,
                task=task,
                language=language,
            )
        else:
            language = language or self.tokenizer.language_code
            task = task or self.tokenizer.task
            if task != self.tokenizer.task or language != self.tokenizer.language_code:
                self.tokenizer = Tokenizer(
                    self.model.hf_tokenizer,
                    self.model.model.is_multilingual,
                    task=task,
                    language=language,
                )

        if self.suppress_numerals:
            previous_suppress_tokens = self.options.suppress_tokens
            numeral_symbol_tokens = find_numeral_symbol_tokens(self.tokenizer)
            logger.info("Suppressing numeral and symbol tokens")
            new_suppressed_tokens = numeral_symbol_tokens + self.options.suppress_tokens
            new_suppressed_tokens = list(set(new_suppressed_tokens))
            self.options = replace(self.options, suppress_tokens=new_suppressed_tokens)

        segments: List[SingleSegment] = []
        total_segments = len(vad_segments)

        # Warn if batch_size is large relative to the number of VAD chunks
        if batch_size >= total_segments:
            warnings.warn(
                f"batch_size ({batch_size}) is >= total VAD chunks ({total_segments}). "
                "Each stream will receive ≤1 segment so rolling context has no effect. "
                "Consider reducing --batch_size.",
                UserWarning,
                stacklevel=2,
            )
        elif batch_size >= total_segments // 2:
            warnings.warn(
                f"batch_size ({batch_size}) is more than half the total VAD chunks ({total_segments}). "
                "Streams will have very few segments; consider reducing --batch_size for better context.",
                UserWarning,
                stacklevel=2,
            )

        if self._use_interleaved_context:
            num_streams = batch_size
            # Distribute segments into streams
            k, m = divmod(len(vad_segments), num_streams)
            stream_segments = []
            start_idx = 0
            for i in range(num_streams):
                part_len = k + 1 if i < m else k
                stream_segments.append(vad_segments[start_idx : start_idx + part_len])
                start_idx += part_len

            # Interleave streams so each batch contains one segment per stream
            interleaved_segments = []
            max_len = max(len(s) for s in stream_segments)
            for i in range(max_len):
                for stream in stream_segments:
                    if i < len(stream):
                        interleaved_segments.append(stream[i])

            vad_segments = interleaved_segments

        for idx, out in enumerate(self.__call__(data(audio, vad_segments), batch_size=batch_size, num_workers=num_workers)):
            if print_progress:
                base_progress = ((idx + 1) / total_segments) * 100
                percent_complete = base_progress / 2 if combined_progress else base_progress
                print(f"Progress: {percent_complete:.2f}%...")
            if progress_callback is not None:
                progress_callback(((idx + 1) / total_segments) * 100)

            text = out['text']
            avg_logprob = out['avg_logprob']
            if isinstance(text, list):
                text = text[0]
            if isinstance(avg_logprob, list):
                avg_logprob = avg_logprob[0]
            
            if verbose:
                print(f"Transcript: [{round(vad_segments[idx]['start'], 3)} --> {round(vad_segments[idx]['end'], 3)}] {text}")
            segments.append(
                {
                "text": text,
                "start": round(vad_segments[idx]['start'], 3),
                "end": round(vad_segments[idx]['end'], 3),
                "avg_logprob": avg_logprob,
                }
            )


        if interleaved_context and batch_size > 1:
            # After first pass, self.previous_batch_context_tokens contains accumulated context
            # from multiple segments of each stream. Use this for proper wrap-around.
            accumulated_context = [list(t) for t in self.previous_batch_context_tokens[:batch_size]]
            # Prepare context for the wrap-around re-run:
            # Stream 0 stays empty (very start of audio)
            # Stream i gets context from Stream i-1's accumulated context
            new_rerun_context = [[] for _ in range(batch_size)]
            for i in range(1, batch_size):
                new_rerun_context[i] = accumulated_context[i - 1]
            # Temporarily overwrite previous_batch_context_tokens for the re-run
            self.previous_batch_context_tokens = new_rerun_context
            first_batch_segments = vad_segments[:batch_size]

            # Reset last_segment_tokens_per_stream and segment index counter for interleaved context pass
            self.last_segment_tokens_per_stream = [[] for _ in range(batch_size)]
            self.stream_segment_indices = [0 for _ in range(batch_size)]

            # Runs the model again just on 'first_batch_segments'
            for i, out in enumerate(self.__call__(data(audio, first_batch_segments), batch_size=batch_size, num_workers=num_workers)):
                text = out['text']
                # Overwrite the existing text with the new wrap-around text
                if isinstance(text, list):
                    text = text[0]
                if verbose:
                    logger.info(f"[INTERLEAVED] Segment {i} context redo:     {text[:80]}...")
                segments[i]['text'] = text
        # Sort segments by start time to restore original order
        segments.sort(key=lambda x: x['start'])

        # revert the tokenizer if multilingual inference is enabled
        if self.preset_language is None:
            self.tokenizer = None

        # revert suppressed tokens if suppress_numerals is enabled
        if self.suppress_numerals:
            self.options = replace(self.options, suppress_tokens=previous_suppress_tokens)

        return {"segments": segments, "language": language}

    def detect_language(self, audio: np.ndarray) -> str:
        if audio.shape[0] < N_SAMPLES:
            logger.warning("Audio is shorter than 30s, language detection may be inaccurate")
        model_n_mels = self.model.feat_kwargs.get("feature_size")
        segment = log_mel_spectrogram(audio[: N_SAMPLES],
                                      n_mels=model_n_mels if model_n_mels is not None else 80,
                                      padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0])
        encoder_output = self.model.encode(segment)
        results = self.model.model.detect_language(encoder_output)
        language_token, language_probability = results[0][0]
        language = language_token[2:-2]
        logger.info(f"Detected language: {language} ({language_probability:.2f}) in first 30s of audio")
        return language


def load_model(
    whisper_arch: str,
    device: str,
    device_index=0,
    compute_type="default",
    asr_options: Optional[dict] = None,
    language: Optional[str] = None,
    vad_model: Optional[Vad]= None,
    vad_method: Optional[str] = "pyannote",
    vad_options: Optional[dict] = None,
    model: Optional[WhisperModel] = None,
    task="transcribe",
    download_root: Optional[str] = None,
    local_files_only=False,
    threads=4,
    use_auth_token: Optional[Union[str, bool]] = None,
) -> FasterWhisperPipeline:
    """Load a Whisper model for inference.
    Args:
        whisper_arch - The name of the Whisper model to load.
        device - The device to load the model on.
        compute_type - The compute type to use for the model.
            Use "default" to automatically select based on device (float16 for GPU, float32 for CPU).
        vad_model - The vad model to manually assign.
        vad_method - The vad method to use. vad_model has a higher priority if it is not None.
        options - A dictionary of options to use for the model.
        language - The language of the model. (use English for now)
        model - The WhisperModel instance to use.
        download_root - The root directory to download the model to.
        local_files_only - If `True`, avoid downloading the file and return the path to the local cached file if it exists.
        threads - The number of cpu threads to use per worker, e.g. will be multiplied by num workers.
    Returns:
        A Whisper pipeline.
    """

    if compute_type == "default":
        compute_type = "float16" if device == "cuda" else "float32"
        logger.info(f"Compute type not specified, defaulting to {compute_type} for device {device}")

    if whisper_arch.endswith(".en"):
        language = "en"

    model = model or WhisperModel(whisper_arch,
                         device=device,
                         device_index=device_index,
                         compute_type=compute_type,
                         download_root=download_root,
                         local_files_only=local_files_only,
                         cpu_threads=threads,
                         use_auth_token=use_auth_token)
    if language is not None:
        tokenizer = Tokenizer(model.hf_tokenizer, model.model.is_multilingual, task=task, language=language)
    else:
        logger.info("No language specified, language will be detected for each audio file (increases inference time)")
        tokenizer = None

    default_asr_options =  {
        "beam_size": 5,
        "best_of": 5,
        "patience": 1,
        "length_penalty": 1,
        "repetition_penalty": 1,
        "no_repeat_ngram_size": 0,
        "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False,
        "prompt_reset_on_temperature": 0.5,
        "initial_prompt": None,
        "prefix": None,
        "suppress_blank": True,
        "suppress_tokens": [-1],
        "without_timestamps": True,
        "max_initial_timestamp": 0.0,
        "word_timestamps": False,
        "prepend_punctuations": "\"'“¿([{-",
        "append_punctuations": "\"'.。,，!！?？:：”)]}、",
        "multilingual": model.model.is_multilingual,
        "suppress_numerals": False,
        "max_new_tokens": None,
        "clip_timestamps": None,
        "hallucination_silence_threshold": None,
        "hotwords": None,
    }

    if asr_options is not None:
        default_asr_options.update(asr_options)

    suppress_numerals = default_asr_options["suppress_numerals"]
    del default_asr_options["suppress_numerals"]

    default_asr_options = TranscriptionOptions(**default_asr_options)

    default_vad_options = {
        "chunk_size": 30, # needed by silero since binarization happens before merge_chunks
        "vad_onset": 0.500,
        "vad_offset": 0.363
    }

    if vad_options is not None:
        default_vad_options.update(vad_options)

    # Note: manually assigned vad_model has higher priority than vad_method!
    if vad_model is not None:
        print("Use manually assigned vad_model. vad_method is ignored.")
        vad_model = vad_model
    else:
        if vad_method == "silero":
            vad_model = Silero(**default_vad_options)
        elif vad_method == "pyannote":
            if device == 'cuda':
                device_vad = f'cuda:{device_index}'
            else:
                device_vad = device
            vad_model = Pyannote(torch.device(device_vad), token=None, **default_vad_options)
        else:
            raise ValueError(f"Invalid vad_method: {vad_method}")

    return FasterWhisperPipeline(
        model=model,
        vad=vad_model,
        options=default_asr_options,
        tokenizer=tokenizer,
        language=language,
        suppress_numerals=suppress_numerals,
        vad_params=default_vad_options,
    )
