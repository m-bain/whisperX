from whisperx.vads.pyannote import Pyannote as Pyannote
from whisperx.vads.silero import Silero as Silero
from whisperx.vads.vad import Vad as Vad

# Add load_vad_model for compatibility
def load_vad_model(model_name="silero", device="cuda", **kwargs):
    """Load VAD model by name."""
    if model_name == "silero":
        return Silero(device=device, **kwargs)
    elif model_name == "pyannote":
        return Pyannote(device=device, **kwargs)
    else:
        raise ValueError(f"Unknown VAD model: {model_name}")
