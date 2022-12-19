<h1 align="center">WhisperX</h1>
<h6 align="center">Made by Max Bain • :globe_with_meridians: <a href="https://www.maxbain.com/">https://www.maxbain.com/</a></h6>

<p align="left">Whisper-Based Automatic Speech Recognition (ASR) with improved timestamp accuracy using forced alignment.

</p>


<h2 align="left">What is it 🔎</h2>

This repository refines the timestamps of openAI's Whisper model via forced aligment with phoneme-based ASR models (e.g. wav2vec2.0) 


**Whisper** is an ASR model [developed by OpenAI](https://github.com/openai/whisper), trained on a large dataset of diverse audio. Whilst it does produces highly accurate transcriptions, the corresponding timestamps are at the utterance-level, not per word, and can be inaccurate by several seconds.

**Phoneme-Based ASR** A suite of models finetuned to recognise the smallest unit of speech distinguishing one word from another, e.g. the element p in "tap". A popular example model is [wav2vec2.0](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self).

**Forced Alignment** refers to the process by which orthographic transcriptions are aligned to audio recordings to automatically generate phone level segmentation.

<img width="1216" align="center" alt="whisperx-arch" src="https://user-images.githubusercontent.com/36994049/208313881-903ab3ea-4932-45fd-b3dc-70876cddaaa2.png">


<h2 align="left">Setup ⚙️</h2>
Install this package using

`pip install git+https://github.com/m-bain/whisperx.git`

You may also need to install ffmpeg, rust etc. Follow openAI instructions here https://github.com/openai/whisper#setup.

<h2 align="left">Examples💬</h2>

### English
Run whisper on example segment (using default params)

`whisperx examples/sample01.wav --model medium.en --output examples/whisperx --align_model WAV2VEC2_ASR_LARGE_LV60K_960H --align_extend 2`

If low gpu memory is required, use a smaller align model e.g. `WAV2VEC2_ASR_BASE_LV60K_960H`

Using normal whisper out of the box, many transcriptions are out of sync:

https://user-images.githubusercontent.com/36994049/207743923-b4f0d537-29ae-4be2-b404-bb941db73652.mov

Now, using *WhisperX* with forced alignment to wav2vec2.0:

https://user-images.githubusercontent.com/36994049/208253969-7e35fe2a-7541-434a-ae91-8e919540555d.mp4


## Other Languages

For non-english ASR, it is best to use the `large` whisper model.

### French
`whisperx --model large --language fr examples/sample_fr_01.wav --align_model VOXPOPULI_ASR_BASE_10K_FR --output_dir examples/whisperx/ --align_extend 2`


https://user-images.githubusercontent.com/36994049/208298804-31c49d6f-6787-444e-a53f-e93c52706752.mov



### German
`whisperx --model large --language de examples/sample_de_01.wav --align_model VOXPOPULI_ASR_BASE_10K_DE --output_dir examples/whisperx/ --align_extend 2`



https://user-images.githubusercontent.com/36994049/208298811-e36002ba-3698-4731-97d4-0aebd07e0eb3.mov




### Italian
`whisperx --model large --language it examples/sample_it_01.wav --align_model VOXPOPULI_ASR_BASE_10K_IT --output_dir examples/whisperx/ --align_extend 2`



https://user-images.githubusercontent.com/36994049/208298819-6f462b2c-8cae-4c54-b8e1-90855794efc7.mov

### Japanese
`whisperx --model large --language ja examples/sample_ja_01.mp3  --align_model wav2vec2-large-xlsr-53-japanese--output_dir examples/whisperx --align_extend 2`




https://user-images.githubusercontent.com/19920981/208448405-60f80c0e-2715-42d8-9437-e19e6362b638.mov


<h2 align="left">Limitations ⚠️</h2>

- Not thoroughly tested, especially for non-english, results may vary -- please post issue to let me know its results on your data
- Whisper normalises spoken numbers e.g. "fifty seven" to arabic numerals "57". Need to perform this normalization after alignment, so the phonemes can be aligned. Currently just ignores numbers.
- Assumes the initial whisper timestamps are accurate to some degree (within margin of 2 seconds, adjust if needed -- bigger margins more prone to alignment errors)
- Hacked this up quite quickly, there might be some errors, please raise an issue if you encounter any.

<h2 align="left">Coming Soon 🗓</h2>

[x] Multilingual init

[x] Subtitle .ass output

[ ] Automatic align model selection based on language detection

[ ] Reduce GPU (clear cache etc.)

[ ] Incorporating word-level speaker diarization

[ ] Inference speedup with batch processing

<h2 align="left">Contact</h2>

Contact maxbain[at]robots[dot]ox[dot]ac[dot]uk if using this for commerical purposes.


<h2 align="left">Acknowledgements 🙏</h2>

Of course, this is mostly just a modification to [openAI's whisper](https://github.com/openai/whisper).
As well as accreditation to this [PyTorch tutorial on forced alignment](https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html)


<h2 align="left">Citation</h2>
If you use this in your research, just cite the repo,

```bibtex
@misc{bain2022whisperx,
  author = {Bain, Max},
  title = {WhisperX},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/m-bain/whisperX}},
}
```

as well as the whisper paper,

```bibtex
@article{radford2022robust,
  title={Robust speech recognition via large-scale weak supervision},
  author={Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2212.04356},
  year={2022}
}
```
and any alignment model used, e.g. wav2vec2.0.

```bibtex
@article{baevski2020wav2vec,
  title={wav2vec 2.0: A framework for self-supervised learning of speech representations},
  author={Baevski, Alexei and Zhou, Yuhao and Mohamed, Abdelrahman and Auli, Michael},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={12449--12460},
  year={2020}
}
```
