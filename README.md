<h1 align="center">WhisperX</h1>

<p align="center">Whisper Automatic Speech Recognition with improved timestamp accuracy using forced alignment.

</p>


<h2 align="center">What is it</h2>

This repository refines the timestamps of openAI's Whisper model via forced aligment with phoneme-level ASR models (e.g. wav2vec2) 


**Whisper** is an Automatic Speech Recognition model [developed by OpenAI](https://github.com/openai/whisper), trained on a large dataset of diverse audio. Whilst it does produces highly accurate transcriptions, the corresponding timestamps are at the utterance-level, not per word, and can be inaccurate by several seconds.

**Forced Alignment** refers to the process by which orthographic transcriptions are aligned to audio recordings to automatically generate phone level segmentation.

<h2 align="center">Setup</h2>
Install this package using

`pip install git+https://github.com/m-bain/whisperx.git`

You may also need to install ffmpeg, rust etc. Follow openAI instructions here https://github.com/openai/whisper#setup.

<h2 align="center">Example</h2>

Run whisper on example segment (using default params)

`whisperx examples/sample01.wav --model medium.en --output examples/whisperx --align_model WAV2VEC2_ASR_LARGE_LV60K_960H --align_extend 2`

Outputs both word-level, and phrase level.

<h2 align="center">Limitations</h2>

- Hacked this up quite quickly, there might be some errors, please raise an issue if you encounter any.
- Currently only working and tested for ENGLISH language.
- Whisper normalises spoken numbers e.g. "fifty seven" to arabic numerals "57". Need to perform this normalization after alignment, so the phonemes can be aligned. Currently just ignores numbers.
- Assumes the initial whisper timestamps are accurate to some degree (within margin of 2 seconds, adjust if needed -- bigger margins more prone to alignment errors)

<h2 align="center">Contact</h2>

Contact maxbain[at]robots.ox.ac.uk if you are using this at scale.

<h2 align="center">Acknowledgements</h2>

-OpenAI's whisper https://github.com/openai/whisper

-PyTorch forced alignment tutorial https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html
