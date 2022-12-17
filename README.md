<h1 align="center">WhisperX</h1>

<p align="left">Whisper-Based Automatic Speech Recognition (ASR) with improved timestamp accuracy using forced alignment.

</p>


<h2 align="left">What is it 🔎</h2>

This repository refines the timestamps of openAI's Whisper model via forced aligment with phoneme-based ASR models (e.g. wav2vec2.0) 


**Whisper** is an ASR model [developed by OpenAI](https://github.com/openai/whisper), trained on a large dataset of diverse audio. Whilst it does produces highly accurate transcriptions, the corresponding timestamps are at the utterance-level, not per word, and can be inaccurate by several seconds.

**Phoneme-Based ASR** A suite of models finetuned to recognise the smallest unit of speech distinguishing one word from another, e.g. the element p in "tap". A popular example model is [wav2vec2.0](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self).

**Forced Alignment** refers to the process by which orthographic transcriptions are aligned to audio recordings to automatically generate phone level segmentation.

<h2 align="left">Setup ⚙️</h2>
Install this package using

`pip install git+https://github.com/m-bain/whisperx.git`

You may also need to install ffmpeg, rust etc. Follow openAI instructions here https://github.com/openai/whisper#setup.

<h2 align="left">Example</h2>

Run whisper on example segment (using default params)

`whisperx examples/sample01.wav --model medium.en --output examples/whisperx --align_model WAV2VEC2_ASR_LARGE_LV60K_960H --align_extend 2`

If the speech is non-english, select an alternative ASR phoneme model from this list https://pytorch.org/audio/stable/pipelines.html#id14



### Qualitative Results:

Using normal whisper out of the box, many transcriptions are out of sync:

https://user-images.githubusercontent.com/36994049/207743923-b4f0d537-29ae-4be2-b404-bb941db73652.mov

Now, using *WhisperX* with forced alignment to wav2vec2.0:

(a) refining segment timestamps

https://user-images.githubusercontent.com/36994049/207744049-5c0ec593-5c68-44de-805b-b1701d6cc968.mov

(b) word-level timestamps

https://user-images.githubusercontent.com/36994049/207744104-ff4faca1-1bb8-41c9-84fe-033f877e5276.mov


<h2 align="left">Limitations ⚠️</h2>

- Currently only tested for ENGLISH language. Check 
- Whisper normalises spoken numbers e.g. "fifty seven" to arabic numerals "57". Need to perform this normalization after alignment, so the phonemes can be aligned. Currently just ignores numbers.
- Assumes the initial whisper timestamps are accurate to some degree (within margin of 2 seconds, adjust if needed -- bigger margins more prone to alignment errors)
- Hacked this up quite quickly, there might be some errors, please raise an issue if you encounter any.

<h2 align="left">Coming Soon 🗓</h2>

[ ] Incorporating word-level speaker diarization

[ ] Inference speedup with batch processing

<h2 align="left">Contact</h2>

Contact maxbain[at]robots.ox.ac.uk non-bug related queries.

<h2 align="left">Acknowledgements 🙏</h2>

Of course, this is mostly just a modification to [openAI's whisper](https://github.com/openai/whisper).
As well as accreditation to this [PyTorch tutorial on forced alignment](https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html)
