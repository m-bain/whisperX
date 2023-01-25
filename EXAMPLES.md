# More Examples

## Other Languages

For non-english ASR, it is best to use the `large` whisper model. Alignment models are automatically picked by the chosen language from the default [lists](https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py#L18).

Currently support default models tested for {en, fr, de, es, it, ja, zh, nl}


If the detected language is not in this list, you need to find a phoneme-based ASR model from [huggingface model hub](https://huggingface.co/models) and test it on your data.

### French
    whisperx --model large --language fr examples/sample_fr_01.wav


https://user-images.githubusercontent.com/36994049/208298804-31c49d6f-6787-444e-a53f-e93c52706752.mov


### German
    whisperx --model large --language de examples/sample_de_01.wav


https://user-images.githubusercontent.com/36994049/208298811-e36002ba-3698-4731-97d4-0aebd07e0eb3.mov


### Italian
    whisperx --model large --language de examples/sample_it_01.wav


https://user-images.githubusercontent.com/36994049/208298819-6f462b2c-8cae-4c54-b8e1-90855794efc7.mov


### Japanese
    whisperx --model large --language ja examples/sample_ja_01.wav


https://user-images.githubusercontent.com/19920981/208731743-311f2360-b73b-4c60-809d-aaf3cd7e06f4.mov
