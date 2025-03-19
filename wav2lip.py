import os
import sys
import numpy as np
from comfy import model_management
from comfy import utils as comfy_utils
import torch
import io
import tempfile
import torchaudio
from pathlib import Path
import subprocess
import hashlib

# Removed unused imports: pydub, soundfile, subprocess

def find_folder(base_path, folder_name):
    for root, dirs, files in os.walk(base_path):
        if folder_name in dirs:
            return Path(root) / folder_name
    return None

def check_model_in_folder(folder_path, model_file):
    model_path = folder_path / model_file
    return model_path.exists(), model_path

base_dir = Path(__file__).resolve().parent

print(f"Base directory: {base_dir}")

# 延迟到process方法中加载模型
wav2lip_model_file = "wav2lip_gan.pth"


current_dir = Path(__file__).resolve().parent
wav2lip_path = current_dir / "wav2lip"
if str(wav2lip_path) not in sys.path:
    sys.path.append(str(wav2lip_path))
print(f"Wav2Lip path added to sys.path: {wav2lip_path}")

def setup_directory(base_dir, dir_name):
    dir_path = os.path.join(base_dir, dir_name)
    os.makedirs(dir_path, exist_ok=True)
    print(f"Directory created or exists: {dir_path}")

setup_directory(base_dir, "facedetection")

current_dir = os.path.dirname(os.path.abspath(__file__))
wav2lip_path = os.path.join(current_dir, "wav2lip")
sys.path.append(wav2lip_path)
print(f"Current directory: {current_dir}")
print(f"Wav2Lip path: {wav2lip_path}")

from .Wav2Lip.wav2lip_node import wav2lip_

# Removed process_audio, get_ffmpeg_path, get_audio, validate_path, hash_path functions

import hashlib
import folder_paths  # Assuming folder_paths is a module you have for handling paths

class LoadAudio:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = folder_paths.filter_files_content_types(os.listdir(input_dir), ["audio", "video"])
        return {
            "required": {
                "audio": (sorted(files), {"audio_upload": True})
            }
        }

    CATEGORY = "audio"

    RETURN_TYPES = ("AUDIO", )
    FUNCTION = "load"

    def load(self, audio):
        audio_path = folder_paths.get_annotated_filepath(audio)
        waveform, sample_rate = torchaudio.load(audio_path)
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        return (audio, )

    @classmethod
    def IS_CHANGED(cls, audio):
        audio_path = folder_paths.get_annotated_filepath(audio)
        m = hashlib.sha256()
        with open(audio_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, audio):
        if not folder_paths.exists_annotated_filepath(audio):
            return f"Invalid audio file: {audio}"
        return True

class Wav2LipTrain:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_path": ("FOLDER", {"description": "训练数据集路径", "folder_path": "./datasets"}),
                "pretrained_model": ("MODEL", {"default": "", "description": "预训练模型路径（可选）"}),
                "batch_size": ("INT", {"default": 16, "min": 1, "max": 128}),
                "learning_rate": ("FLOAT", {"default": 1e-4, "min": 1e-6, "max": 1e-2}),
                "max_epochs": ("INT", {"default": 100, "min": 1}),
                "checkpoint_interval": ("INT", {"default": 10, "min": 1}),
                "freeze_layers": ("STRING", {"default": "encoder.*", "description": "冻结层正则表达式"})
            }
        }

    CATEGORY = "ComfyUI/Wav2Lip"

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "train"

    def train(self, dataset_path, batch_size, learning_rate, max_epochs, checkpoint_interval):
        import torch.optim as optim
        from torch.utils.data import DataLoader
        
        # 初始化模型
        model = Wav2Lip().to(device)
        
        # 加载预训练权重
        if pretrained_model:
            checkpoint = torch.load(pretrained_model)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 冻结指定层
            for name, param in model.named_parameters():
                if re.match(freeze_layers, name):
                    param.requires_grad = False
        
        # 只优化需要梯度的参数
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(trainable_params, lr=learning_rate)
        
        from .Wav2Lip.wav2lip_node import Wav2Lip
        from torch.utils.data import Dataset, DataLoader
        import torch.nn as nn
        
        # 创建checkpoints目录
        checkpoints_path = os.path.join(base_dir, 'checkpoints')
        os.makedirs(checkpoints_path, exist_ok=True)
        
        # 自定义数据集类
        class LipSyncDataset(Dataset):
            def __init__(self, dataset_path):
                self.data = []  # 实现实际的数据加载逻辑
                
            def __len__(self):
                return len(self.data)
                
            def __getitem__(self, idx):
                return self.data[idx]
        
        # 初始化模型和优化器
        model = Wav2Lip().to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.L1Loss()
        
        # 加载数据集
        train_dataset = LipSyncDataset(dataset_path)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 训练循环
        for epoch in range(max_epochs):
            model.train()
            total_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                # 实现实际的前向传播和损失计算
                loss = criterion(None, None)  # 替换实际计算
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            # 保存检查点
            if epoch % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoints_path, f'wav2lip_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss/len(train_loader)
                }, checkpoint_path)
        
        return (model,)

class Wav2Lip:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mode": (["sequential", "repetitive"], {"default": "sequential"}),
                "face_detect_batch": ("INT", {"default": 8, "min": 1, "max": 100}),
                "padding": ("STRING", {"default": "0,25,0,0", "description": "Padding values in 'top,bottom,left,right' format (integer values)"}),
                "audio": ("AUDIO", ),
                "model_type": (["wav2lip_gan", "wav2lip"], {"default": "wav2lip_gan"})
            },
        }

    CATEGORY = "ComfyUI/Wav2Lip"

    RETURN_TYPES = ("IMAGE", "AUDIO", )
    RETURN_NAMES = ("images", "audio", )
    FUNCTION = "process"

    def process(self, images, mode, face_detect_batch, padding, audio, model_type):
        try:
            pad_top, pad_bottom, pad_left, pad_right = map(int, padding.split(','))
            if not all(isinstance(x, int) for x in (pad_top, pad_bottom, pad_left, pad_right)):
                raise ValueError
        except:
            raise ValueError("Invalid padding format. Use 'top,bottom,left,right' with 0-100 integers")
        # 运行时加载模型
        if not hasattr(self, 'last_model_type') or model_type != self.last_model_type:
            checkpoints_path = find_folder(base_dir, "checkpoints")
            model_filename = f"{model_type}.pth"
            model_exists, model_path = check_model_in_folder(checkpoints_path, model_filename)
            assert model_exists, f"Model {model_filename} not found in {checkpoints_path}"
            self.last_model_type = model_type
            self.last_model_path = model_path
        else:
            model_path = self.last_model_path

        in_img_list = []
        for i in images:
            in_img = i.numpy().squeeze()
            in_img = (in_img * 255).astype(np.uint8)
            in_img_list.append(in_img)

        if audio is None or "waveform" not in audio or "sample_rate" not in audio:
            raise ValueError("Valid audio input is required.")

        waveform = audio["waveform"].squeeze(0).numpy()  # Expected shape: [channels, samples]
        sample_rate = audio["sample_rate"]

        # Step 1: Convert to Mono if Necessary
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            # Average the channels to convert to mono
            waveform = waveform.mean(axis=0)
            print(f"Converted multi-channel audio to mono. New shape: {waveform.shape}")
        elif waveform.ndim == 2 and waveform.shape[0] == 1:
            # Already mono, remove the channel dimension
            waveform = waveform.squeeze(0)
            print(f"Audio is already mono. Shape: {waveform.shape}")
        elif waveform.ndim != 1:
            raise ValueError(f"Unsupported waveform shape: {waveform.shape}")

        # Step 2: Ensure the Sample Rate is 16000 Hz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform_tensor = torch.tensor(waveform)
            waveform = resampler(waveform_tensor).numpy()
            sample_rate = 16000
            print(f"Resampled audio to {sample_rate} Hz.")

        # Step 3: Normalize the Waveform to [-1, 1]
        waveform = waveform.astype(np.float32)
        max_val = np.abs(waveform).max()
        if max_val > 0:
            waveform /= max_val
        print(f"Normalized waveform. Max value after normalization: {np.abs(waveform).max()}")

        # Step 4: Save the Waveform to a Temporary WAV File
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio_path = temp_audio.name
            # Convert waveform back to tensor and ensure it's 2D [1, samples]
            waveform_tensor = torch.tensor(waveform).unsqueeze(0)  # Shape: [1, samples]
            torchaudio.save(temp_audio_path, waveform_tensor, sample_rate)
            print(f"Saved temporary audio file at {temp_audio_path}")

        try:
            # Process with Wav2Lip model
            out_img_list = wav2lip_(in_img_list, temp_audio_path, face_detect_batch, mode, model_path, pad_top, pad_bottom, pad_left, pad_right)
        finally:
            os.unlink(temp_audio_path)  # Ensure temporary file is deleted
            print(f"Deleted temporary audio file at {temp_audio_path}")

        out_tensor_list = []
        for out_img in out_img_list:
            out_img = out_img.astype(np.float32) / 255.0
            out_tensor = torch.from_numpy(out_img)
            out_tensor_list.append(out_tensor)

        images = torch.stack(out_tensor_list, dim=0)

        # Return the processed images and the original audio
        return (images, audio,)


NODE_CLASS_MAPPINGS = {
    "Wav2Lip": Wav2Lip,
    "LoadAudio": LoadAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wav2Lip": "Wav2Lip",
    "LoadAudio": "Load Audio",
}
