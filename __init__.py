from .wav2lip import Wav2Lip, LoadAudio, Wav2LipTrain

NODE_CLASS_MAPPINGS = {
    "Wav2Lip": Wav2Lip,
    "LoadAudio": LoadAudio,
    "Wav2LipTrain": Wav2LipTrain
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wav2Lip": "Wav2Lip 推理",
    "LoadAudio": "Load Audio",
    "Wav2LipTrain": "Wav2Lip 训练",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
