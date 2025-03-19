from .wav2lip import Wav2Lip, LoadAudio

NODE_CLASS_MAPPINGS = {
    "Wav2Lip": Wav2Lip,
    "LoadAudio": LoadAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wav2Lip": "Wav2Lip 推理",
    "LoadAudio": "Load Audio",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
