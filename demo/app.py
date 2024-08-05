# gradio app.py
import whisperx
import logging
import tempfile
from copy import deepcopy
import gradio as gr
import torch
import torchaudio
from typing import Tuple

formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"  # noqa
logging.basicConfig(format=formatter, level=logging.INFO)

LANGUAGES = ["English", "汉语"]
LANGUAGES_MAP = {"English": "en", "汉语": "zh"}

if torch.cuda.is_available():
    device = "cuda:0"
    compute_type = "float16"
    logging.info(f"Running inference on the GPU in {compute_type}.")
else:
    device = "cpu"
    compute_type = "float32"
    logging.info(f"Running inference on the CPU in {compute_type}.")

model_map = {
    "large-v2": whisperx.load_model("large-v2",
                                    device,
                                    compute_type=compute_type),
    # "small": whisperx.load_model("small",
    #                                 device,
    #                                 compute_type=compute_type),
}


def asr(
    audio_file: str,
    initial_prompt: str,
    model_name: str,
    language: str,
    disable_vad: bool,
    batch_size: int,
    temperature: float = 0.0,
    beam_size: int = 5,
):
    model = model_map[model_name]

    options = deepcopy(model.options)
    if initial_prompt:
        options._replace("initial_prompt", initial_prompt)
        model.options = options

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio,
                              batch_size=batch_size,
                              language=LANGUAGES_MAP[language],
                              disable_vad=disable_vad,
                              print_progress=False)
    
    model.options = options

    if not result or not result["segments"]:
        return "No result"
    return f"language: {result['language']} :\n" + " ".join([seg["text"] for seg in result["segments"]])


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_audio = gr.Audio(sources=["upload", "microphone"],
                                type="filepath",
                                format="wav",
                                label="Input Audio: upload wav file")
            with gr.Row():
                model_radio = gr.Radio(label="Model", choices=list(model_map.keys()), value="large-v2")
                language_radio = gr.Radio(label="Language", choices=LANGUAGES, value="English")
                disable_vad = gr.Checkbox(label="Disable VAD(audio <=30 seconds)", value=False)
            with gr.Column():
                batch_size_slider = gr.Slider(4, 16, 4, label="Batch Size [4, 16]", step=4)
                temperature_slider = gr.Slider(0.0, 1.0, 0.0, label="Temperature (0.0, 1.0)")
                beam_size_slider = gr.Slider(1, 10, 5, label="Beam Size [1, 10]", step=1)

            submit_btn = gr.Button("Submit")

        with gr.Column():
            prompt_text = gr.Textbox(label="Text Prompt(If empty, will not prompt the model)",
                                    value="",
                                    lines=1,
                                    max_lines=4)
            outputs = gr.Textbox(label="Transcription",
                                    type="text",
                                    lines=4,
                                    max_lines=10,
                                    show_copy_button=True)

    inputs = [input_audio, prompt_text, model_radio, language_radio, disable_vad, batch_size_slider, temperature_slider, beam_size_slider]
    submit_btn.click(fn=asr, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", ssl_verify=False, share=False)
