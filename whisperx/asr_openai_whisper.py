from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import numpy as np
import os
from typing import List, Optional, Union, Dict, Any, Tuple
from dataclasses import dataclass, field

from whisperx.audio import N_SAMPLES, SAMPLE_RATE, load_audio, log_mel_spectrogram
from whisperx.types import SingleSegment, TranscriptionResult
from whisperx.vads import Vad, Silero, Pyannote, SileroCustom
from whisperx.utils import LANGUAGES, TO_LANGUAGE_CODE


@dataclass
class OpenAIWhisperOptions:
    """Options for OpenAI Whisper transcription"""
    # Keep only options supported by Hugging Face Whisper model
    num_beams: int = 5
    temperature: float = 0.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    length_penalty: float = 1.0
    suppress_tokens: List[int] = field(default_factory=lambda: [-1])
    without_timestamps: bool = True
    return_token_probabilities: bool = False
    max_new_tokens: Optional[int] = None


def find_numeral_symbol_tokens(tokenizer):
    """Find tokens that contain numerals or currency symbols."""
    numeral_symbol_tokens = []
    for token_id in range(len(tokenizer.get_vocab())):
        token = tokenizer.convert_ids_to_tokens(token_id)
        if isinstance(token, str):
            token = token.replace("▁", "")  # Remove sentencepiece prefix
            has_numeral_symbol = any(c in "0123456789%$£" for c in token)
            if has_numeral_symbol:
                numeral_symbol_tokens.append(token_id)
    return numeral_symbol_tokens


class OpenAIWhisperPipeline:
    """
    Hugging Face Pipeline wrapper for OpenAI Whisper model.
    """
    def __init__(
        self,
        model,
        processor,
        vad_model,
        vad_params: dict,
        options: OpenAIWhisperOptions,
        device: Union[int, str, torch.device] = -1,
        language: Optional[str] = None,
        suppress_numerals: bool = False,
    ):
        self.model = model
        self.processor = processor
        self.options = options
        self.preset_language = language
        self.suppress_numerals = suppress_numerals
        
        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif device < 0:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device}")
            
        self.model.to(self.device)
        # Determine the model's dtype
        self.dtype = next(self.model.parameters()).dtype
        
        self.vad_model = vad_model
        self._vad_params = vad_params
        
        if language:
            self.language_token = f"<|{language}|>"
        else:
            self.language_token = None
            
        # For suppressing numeral tokens if needed
        if suppress_numerals:
            self.numeral_symbol_tokens = find_numeral_symbol_tokens(self.processor.tokenizer)
        else:
            self.numeral_symbol_tokens = []

    def preprocess_audio(self, audio_data):
        """Preprocess audio data for the model."""
        # The HF WhisperFeatureExtractor uses 80 mel bins by default
        # Access it from the feature_extractor's config
        
        n_mels = 128  if 'v3' in self.model.name_or_path else 80 # Default value for Whisper models
        if hasattr(self.processor, "feature_extractor") and hasattr(self.processor.feature_extractor, "config"):
            n_mels = getattr(self.processor.feature_extractor.config, "num_mel_bins", 80)
        
        features = log_mel_spectrogram(
            audio_data,
            n_mels=n_mels,
            padding=N_SAMPLES - audio_data.shape[0] if audio_data.shape[0] < N_SAMPLES else 0,
        )
        # Convert features to match model's dtype
        return features.to(device=self.device, dtype=self.dtype)

    def detect_language(self, audio: np.ndarray) -> Tuple[str, float]:
        """
        Detect the language of the audio using the Whisper model.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Tuple of (detected language code, probability)
        """
        if self.preset_language is not None:
            return self.preset_language, 1.0
            
        if audio.shape[0] < N_SAMPLES:
            print("Warning: audio is shorter than 30s, language detection may be inaccurate.")
            
        # Take first 30 seconds for language detection
        audio_segment = audio[:N_SAMPLES] if audio.shape[0] >= N_SAMPLES else audio
        
        features = self.preprocess_audio(audio_segment)
        input_features = features.unsqueeze(0)  # Add batch dimension
        
        # Run encoder on features
        with torch.no_grad():
            # Pass through encoder 
            encoder_output = self.model.model.encoder(input_features)
            # Extract hidden states from BaseModelOutput
            encoder_hidden_states = encoder_output.last_hidden_state
            
            # Use start of transcript token as input
            sot_token = self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
            decoder_input = torch.tensor([[sot_token]]).to(self.device)
            
            # Get decoder output
            decoder_output = self.model.model.decoder(
                input_ids=decoder_input,
                encoder_hidden_states=encoder_hidden_states
            )
            
            # Pass decoder hidden states through language model head to get logits
            hidden_states = decoder_output.last_hidden_state
            logits = self.model.proj_out(hidden_states)[:, 0]
            
            # Get language token probabilities
            # Collect all language tokens
            all_language_tokens = []
            all_language_codes = []
            for lang_code, lang_name in LANGUAGES.items():
                token_id = self.processor.tokenizer.convert_tokens_to_ids(f"<|{lang_code}|>")
                if token_id != self.processor.tokenizer.unk_token_id:
                    all_language_tokens.append(token_id)
                    all_language_codes.append(lang_code)
            
            # Create mask to only keep language tokens
            mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=self.device)
            mask[all_language_tokens] = False
            logits[:, mask] = -float('inf')
            
            # Get most probable language token and probabilities
            language_token_id = logits.argmax(dim=-1)[0].item()
            language_probs = torch.softmax(logits, dim=-1)[0].cpu()
            
            # Get the language code from the language token
            if language_token_id in all_language_tokens:
                idx = all_language_tokens.index(language_token_id)
                language = all_language_codes[idx]
                probability = language_probs[language_token_id].item()
                print(f"Detected language: {language} (probability: {probability:.4f})")
                possible_languages = ['en', 'hi', 'ta', 'te', 'kn']
                if language not in possible_languages:
                    print(f"Detected language {language} not in possible languages. Defaulting to English.")
                    language = 'en'
                    probability = 1.0
                return language, round(probability, 3)
            else:
                print(f"Could not detect language, defaulting to English. Token: {language_token_id}")
                return "en", 1.0

    def transcribe_segment(self, audio_segment: np.ndarray, language: str) -> Union[str, Dict[str, Any]]:
        """Transcribe a single audio segment."""
        features = self.preprocess_audio(audio_segment)
        input_features = features.unsqueeze(0)  # Add batch dimension
        # Configure generation parameters from options
        gen_kwargs = {
            "max_new_tokens": self.options.max_new_tokens,
            "num_beams": self.options.num_beams,
            "num_return_sequences": 1,
            "temperature": self.options.temperature,
            "repetition_penalty": self.options.repetition_penalty,
            "no_repeat_ngram_size": self.options.no_repeat_ngram_size,
            "length_penalty": self.options.length_penalty,
            "return_dict_in_generate": self.options.return_token_probabilities,
            "output_scores": self.options.return_token_probabilities,
        }
        
        # Set language and task
        try:
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                task="transcribe", 
                language=language, 
                no_timestamps=self.options.without_timestamps
            )
        except ValueError:
            # If language is not supported, try with English
            print(f"Warning: Language {language} not supported by model. Falling back to English.")
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                task="transcribe", 
                language="en", 
                no_timestamps=self.options.without_timestamps
            )
            language = "en"
        
        # Suppress tokens if needed
        if self.suppress_numerals:
            suppress_tokens = list(set(self.options.suppress_tokens + self.numeral_symbol_tokens))
        else:
            suppress_tokens = self.options.suppress_tokens
            
        if suppress_tokens:
            gen_kwargs["suppress_tokens"] = suppress_tokens
            
        # Generate transcription
        with torch.no_grad():
            outputs = self.model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                **gen_kwargs
            )
            
        # Process output depending on whether probabilities are requested
        if self.options.return_token_probabilities:
            sequences = outputs.sequences
            scores = outputs.scores
        
            # Process sequences and scores to get tokens and probabilities
            # Extract the first sequence (we only generate one)
            sequence = sequences[0].cpu().tolist()
            
            # Remove special tokens for the final text
            transcription = self.processor.decode(sequence, skip_special_tokens=True)
            
            # Process token probabilities
            token_probs = []
            for i, token in enumerate(sequences[0].cpu().tolist()[4:-1]): # Ignore special tokens (4 in beginning and 1 in end)
                decoded_token = self.processor.decode([token])
                probs = torch.nn.functional.softmax(scores[i], dim=-1)
                token_probs.append({
                    "token": decoded_token,
                    "probability": round(probs[0, token].item(), 3)
                })
            
            return {
                "text": transcription.strip(),
                "probabilities": dict([(token["token"], token["probability"]) for token in token_probs])
            }
        else:
            # For regular mode, just decode and return the text
            if isinstance(outputs, torch.Tensor):
                transcription = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            else:
                transcription = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
            return transcription.strip()

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
    ) -> TranscriptionResult:
        """Transcribe audio using OpenAI Whisper model."""
        if isinstance(audio, str):
            audio = load_audio(audio)
            
        # Pre-process audio and merge chunks as defined by the respective VAD child class
        # In case vad_model is manually assigned follow the functionality of pyannote toolkit
        if issubclass(type(self.vad_model), Vad):
            waveform = self.vad_model.preprocess_audio(audio)
            merge_chunks = self.vad_model.merge_chunks
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
        
        # Detect language if not provided
        detected_language = self.detect_language(audio) if not language else (language, 1.0)
        language, probability = detected_language
        
        segments: List[SingleSegment] = []
        total_segments = len(vad_segments)
        
        for idx, segment in enumerate(vad_segments):
            if print_progress:
                base_progress = ((idx + 1) / total_segments) * 100
                percent_complete = base_progress / 2 if combined_progress else base_progress
                print(f"Progress: {percent_complete:.2f}%...")
                
            start_frame = int(segment['start'] * SAMPLE_RATE)
            end_frame = int(segment['end'] * SAMPLE_RATE)
            audio_segment = audio[start_frame:end_frame]
            
            result = self.transcribe_segment(audio_segment, language)
            
            if verbose:
                if isinstance(result, dict):
                    print(f"Transcript: [{round(segment['start'], 3)} --> {round(segment['end'], 3)}] {result['text']}")
                else:
                    print(f"Transcript: [{round(segment['start'], 3)} --> {round(segment['end'], 3)}] {result}")
            
            # Create segment with or without token probabilities
            if isinstance(result, dict):
                segments.append({
                    "text": result["text"],
                    "start": round(segment['start'], 3),
                    "end": round(segment['end'], 3),
                    "probabilities": result["probabilities"]
                })
            else:
                segments.append({
                    "text": result,
                    "start": round(segment['start'], 3),
                    "end": round(segment['end'], 3)
                })
            
        return {"segments": segments, "language": language, "language_probability": probability}


def load_model(
    whisper_arch: str,
    device: str,
    device_index=0,
    compute_type="float16",
    asr_options: Optional[dict] = None,
    language: Optional[str] = None,
    vad_model: Optional[Vad] = None,
    vad_method: Optional[str] = "silero_custom",
    vad_options: Optional[dict] = None,
    task="transcribe",
    download_root: Optional[str] = None,
    local_files_only=False,
    vad_onnx=True,
    silero_merge_cutoff=0.1,
    return_token_probabilities=False
) -> OpenAIWhisperPipeline:
    """Load an OpenAI Whisper model from Hugging Face for inference.
    Args:
        whisper_arch - The name of the Whisper model to load from Hugging Face.
        device - The device to load the model on.
        compute_type - The compute type to use for the model.
        vad_method - The vad method to use. vad_model has higher priority if is not None.
        asr_options - A dictionary of options to use for the model.
        language - The language of the model.
        vad_model - The VAD model instance to use.
        download_root - The root directory to download the model to.
        local_files_only - If `True`, avoid downloading the file and return the path to the local cached file if it exists.
        vad_onnx - If `True`, use the ONNX version of the Silero VAD model.
        silero_merge_cutoff - The merge cutoff for the Silero VAD model.
        return_token_probabilities - Whether to return token probabilities in the output.
    Returns:
        A Whisper pipeline.
    """
    # If model ends with .en, it's an English-only model
    if whisper_arch.endswith(".en"):
        language = "en"
        
    # Set device type
    device_type = "cuda" if "cuda" in device else "cpu"
    dtype = torch.float16 if compute_type == "float16" and device_type == "cuda" else torch.float32
    
    # Load processor and model
    processor = WhisperProcessor.from_pretrained(
        whisper_arch,
        cache_dir=download_root,
        local_files_only=local_files_only
    )
    
    model = WhisperForConditionalGeneration.from_pretrained(
        whisper_arch,
        cache_dir=download_root,
        local_files_only=local_files_only,
        device_map=device if device_type == "cuda" else None,
        torch_dtype=dtype
    )
    
    # Configure default ASR options - include only supported options
    default_asr_options = {
        "num_beams": 5,
        "temperature": 0.0,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "length_penalty": 1.0,
        "suppress_tokens": [-1],
        "without_timestamps": True,
        "return_token_probabilities": return_token_probabilities,
        "max_new_tokens": None,
        "suppress_numerals": False,
    }
    
    if asr_options is not None:
        for key, value in asr_options.items():
            if key in default_asr_options:
                default_asr_options[key] = value
        
    suppress_numerals = default_asr_options.pop("suppress_numerals", False)
    
    # Configure VAD options
    default_vad_options = {
        "chunk_size": 30,  # needed by silero since binarization happens before merge_chunks
        "vad_onset": 0.500,
        "vad_offset": 0.363,
        "vad_onnx": vad_onnx,
        "silero_merge_cutoff": silero_merge_cutoff
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
        elif vad_method == "silero_custom":
            vad_model = SileroCustom(**default_vad_options)
        elif vad_method == "pyannote":
            vad_model = Pyannote(torch.device(device), use_auth_token=None, **default_vad_options)
        else:
            raise ValueError(f"Invalid vad_method: {vad_method}")
            
    options = OpenAIWhisperOptions(**default_asr_options)
    
    return OpenAIWhisperPipeline(
        model=model,
        processor=processor,
        vad_model=vad_model,
        options=options,
        device=device,
        language=language,
        suppress_numerals=suppress_numerals,
        vad_params=default_vad_options,
    )

