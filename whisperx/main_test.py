import whisperx
import gc
import time

device = "cuda"
audio_file = "../../../../49966)_Das_erstaunliche_Leben_des_Walter_Mitty.mp4"
batch_size = 8  # 16 reduce if low on GPU mem
compute_type = "int8"  # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Load model
start = time.time()
model = whisperx.load_model("large-v2", device, compute_type=compute_type)  # large-v2
end = time.time()
print(f"Model loading took {end - start:.2f} seconds")

# 2. Load audio
start = time.time()
audio = whisperx.load_audio(audio_file)
end = time.time()
print(f"Audio loading took {end - start:.2f} seconds")

# 3. Transcribe
start = time.time()
result = model.transcribe(audio, batch_size=batch_size, language="de")
end = time.time()
print(f"Transcription took {end - start:.2f} seconds")
print(result["segments"])  # before alignment

# Free up memory if needed
# gc.collect(); del model

# 4. Load alignment model
start = time.time()
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
end = time.time()
print(f"Alignment model loading took {end - start:.2f} seconds")

# 5. Align
start = time.time()
result = whisperx.align(
    result["segments"],
    model_a,
    metadata,
    audio,
    device,
    return_char_alignments=False,
)
end = time.time()
print(f"Alignment took {end - start:.2f} seconds")
# print(result["segments"])  # after alignment

# Free up memory if needed
# gc.collect(); del model_a

# 6. (Optional) Diarization - add similar timing if used

print("Final segments with speaker IDs (if diarization applied):")
# print(result["segments"])

# import whisperx
# import gc
#
# device = "cpu"
# audio_file = "tests/spider-esp.wav"
# batch_size = 4 # 16 reduce if low on GPU mem
# compute_type = "int8" #16 change to "int8" if low on GPU mem (may reduce accuracy)
#
# # 1. Transcribe with original whisper (batched)
# model = whisperx.load_model("large-v2", device, compute_type=compute_type) #large-v2
#
# # save model to local path (optional)
# # model_dir = "/path/"
# # model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)
#
# audio = whisperx.load_audio(audio_file)
# result = model.transcribe(audio, batch_size=batch_size)
# print(result["segments"]) # before alignment
#
# # delete model if low on GPU resources
# # import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model
#
# # 2. Align whisper output
# model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
# result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
#
# print(result["segments"]) # after alignment
#
# # delete model if low on GPU resources
# # import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model_a
#
# # 3. Assign speaker labels
# # diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)
#
# # add min/max number of speakers if known
# # diarize_segments = diarize_model(audio)
# # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
#
# # result = whisperx.assign_word_speakers(diarize_segments, result)
# # print(diarize_segments)
# print(result["segments"]) # segments are now assigned speaker IDs