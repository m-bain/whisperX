import json
import logging
import os
import time

import librosa
import numpy as np
import torch
from dotenv import load_dotenv
from scipy.io import wavfile

import whisperx
import whisperx.diarize

logger = logging.getLogger(__name__)
load_dotenv()

whisper_model = None
diarize_model = None

align_model = None
language_code = None
align_metadata = None


def save_wav(wav: np.ndarray, output_path: str, sample_rate=24000):
    # wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    wav_norm = wav * 32767
    wavfile.write(output_path, sample_rate, wav_norm.astype(np.int16))


def init_whisperx():
    load_whisper_model()
    load_align_model()
    load_diarize_model()


def load_whisper_model(
    model_name: str = "large-v3", download_root="models/ASR/whisper", device="auto"
):
    if model_name == "large":
        model_name = "large-v3"
    global whisper_model
    if whisper_model is not None:
        return
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading WhisperX model: {model_name}")
    t_start = time.time()
    whisper_model = whisperx.load_model(
        model_name, download_root=download_root, device=device, language="en"
    )
    t_end = time.time()
    logger.info(f"Loaded WhisperX model: {model_name} in {t_end - t_start:.2f}s")


def load_align_model(language="en", device="auto"):
    global align_model, language_code, align_metadata
    if align_model is not None and language_code == language:
        return
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    language_code = language
    t_start = time.time()
    align_model, align_metadata = whisperx.load_align_model(
        language_code=language_code, device=device
    )
    t_end = time.time()
    logger.info(f"Loaded alignment model: {language_code} in {t_end - t_start:.2f}s")


def load_diarize_model(device="auto"):
    global diarize_model
    if diarize_model is not None:
        return
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time.time()
    diarize_model = whisperx.diarize.DiarizationPipeline(
        token=os.getenv("HF_TOKEN"), device=device
    )
    t_end = time.time()
    logger.info(f"Loaded diarization model in {t_end - t_start:.2f}s")


def merge_segments(transcript, ending="!\"').:;?]}~"):
    merged_transcription = []
    buffer_segment = None

    for segment in transcript:
        if buffer_segment is None:
            buffer_segment = segment
        else:
            # Check if the last character of the 'text' field is a punctuation mark
            if buffer_segment["text"][-1] in ending:
                # If it is, add the buffered segment to the merged transcription
                merged_transcription.append(buffer_segment)
                buffer_segment = segment
            else:
                # If it's not, merge this segment with the buffered segment
                buffer_segment["text"] += " " + segment["text"]
                buffer_segment["end"] = segment["end"]

    # Don't forget to add the last buffered segment
    if buffer_segment is not None:
        merged_transcription.append(buffer_segment)

    return merged_transcription


def transcribe_audio(
    folder,
    model_name: str = "large",
    download_root="models/ASR/whisper",
    device="auto",
    batch_size=32,
    diarization=True,
    min_speakers=None,
    max_speakers=None,
):
    logger.info(f"transcribe_audio: {folder}")
    if os.path.exists(os.path.join(folder, "transcript.json")):
        logger.info(f"Transcript already exists in {folder}")
        return True

    wav_path = os.path.join(folder, "audio_vocals.wav")
    if not os.path.exists(wav_path):
        return False

    logger.info(f"Transcribing {wav_path}")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    load_whisper_model(model_name, download_root, device)
    rec_result = whisper_model.transcribe(
        wav_path, batch_size=batch_size, language="en"
    )

    if rec_result["language"] == "nn":
        logger.warning(f"No language detected in {wav_path}")
        return False

    load_align_model(rec_result["language"])
    rec_result = whisperx.align(
        rec_result["segments"],
        align_model,
        align_metadata,
        wav_path,
        device,
        return_char_alignments=False,
    )

    if diarization:
        load_diarize_model(device)
        diarize_segments = diarize_model(
            wav_path, min_speakers=min_speakers, max_speakers=max_speakers
        )
        rec_result = whisperx.assign_word_speakers(diarize_segments, rec_result)

    transcript = [
        {
            "start": segement["start"],
            "end": segement["end"],
            "text": segement["text"].strip(),
            "speaker": segement.get("speaker", "SPEAKER_00"),
        }
        for segement in rec_result["segments"]
    ]
    transcript = merge_segments(transcript)
    with open(os.path.join(folder, "transcript.json"), "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=4, ensure_ascii=False)
    logger.info(
        f"Transcribed {wav_path} successfully, and saved to {os.path.join(folder, 'transcript.json')}"
    )
    generate_speaker_audio(folder, transcript)
    return True


def generate_speaker_audio(folder, transcript):
    wav_path = os.path.join(folder, "audio_vocals.wav")
    audio_data, samplerate = librosa.load(wav_path, sr=24000)
    speaker_dict = dict()
    length = len(audio_data)
    delay = 0.05
    for segment in transcript:
        start = max(0, int((segment["start"] - delay) * samplerate))
        end = min(int((segment["end"] + delay) * samplerate), length)
        speaker_segment_audio = audio_data[start:end]
        speaker_dict[segment["speaker"]] = np.concatenate(
            (
                speaker_dict.get(segment["speaker"], np.zeros((0,))),
                speaker_segment_audio,
            )
        )

    speaker_folder = os.path.join(folder, "SPEAKER")
    if not os.path.exists(speaker_folder):
        os.makedirs(speaker_folder)

    for speaker, audio in speaker_dict.items():
        speaker_file_path = os.path.join(speaker_folder, f"{speaker}.wav")
        save_wav(audio, speaker_file_path)


def transcribe_all_audio_under_folder(
    folder,
    model_name: str = "large",
    download_root="models/ASR/whisper",
    device="auto",
    batch_size=32,
    diarization=True,
    min_speakers=None,
    max_speakers=None,
):
    """
    Input: <folder>/audio_vocals.wav
    Output: <folder>/transcript.json
            <folder>/SPEAKER/<speaker>.wav
    """

    logger.info(f"transcribe_all_audio_under_folder: {folder}")
    for root, dirs, files in os.walk(folder):
        if "audio_vocals.wav" in files and "transcript.json" not in files:
            transcribe_audio(
                root,
                model_name,
                download_root,
                device,
                batch_size,
                diarization,
                min_speakers,
                max_speakers,
            )
    return f"Transcribed all audio under {folder}"


def cli():
    logger.info("============")
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "audio_folder",
        type=str,
        help="audio folder which must contain file named audio_vocals.wav in itself or any subfolder",
    )
    args = parser.parse_args()

    transcribe_all_audio_under_folder(args.audio_folder, "turbo")


if __name__ == "__main__":
    cli()
