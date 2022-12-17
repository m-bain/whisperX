<h1 align="center">WhisperX</h1>
<h6 align="center">Made by Max Bain • :globe_with_meridians: <a href="https://www.maxbain.com/">https://www.maxbain.com/</a></h6>

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

If the speech is non-english, select model from this [list](https://pytorch.org/audio/stable/pipelines.html#id14) that has been trained on desired language.



### Qualitative Results:

Using normal whisper out of the box, many transcriptions are out of sync:

https://user-images.githubusercontent.com/36994049/207743923-b4f0d537-29ae-4be2-b404-bb941db73652.mov

Now, using *WhisperX* with forced alignment to wav2vec2.0:

https://user-images.githubusercontent.com/36994049/208253969-7e35fe2a-7541-434a-ae91-8e919540555d.mp4


<h2 align="left">Limitations ⚠️</h2>

- Currently only tested for _english_ language, results may vary with different languages.
- Whisper normalises spoken numbers e.g. "fifty seven" to arabic numerals "57". Need to perform this normalization after alignment, so the phonemes can be aligned. Currently just ignores numbers.
- Assumes the initial whisper timestamps are accurate to some degree (within margin of 2 seconds, adjust if needed -- bigger margins more prone to alignment errors)
- Hacked this up quite quickly, there might be some errors, please raise an issue if you encounter any.

<h2 align="left">Coming Soon 🗓</h2>

[x] Multilingual init

[x] Subtitle .ass output

[ ] Incorporating word-level speaker diarization

[ ] Inference speedup with batch processing

<h2 align="left">Contact</h2>

Contact maxbain[at]robots[dot]ox[dot]ac[dot]uk if using this for commerical purposes.

<h2 align="left">Acknowledgements 🙏</h2>

Of course, this is mostly just a modification to [openAI's whisper](https://github.com/openai/whisper).
As well as accreditation to this [PyTorch tutorial on forced alignment](https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html)
