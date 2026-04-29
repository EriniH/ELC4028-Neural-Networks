import os
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import random

class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, audio_aug=False, image_aug=False):
        """
        data_dir: Path to 'Train' or 'Test' directory containing .wav files.
        audio_aug: Boolean, if True apply speed +/- 3% and audio noise.
        image_aug: Boolean, if True apply horizontal squeeze/expand +/- 3% and image noise.
        """
        self.data_dir = data_dir
        self.audio_aug = audio_aug
        self.image_aug = image_aug
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
        
        # Spectrogram converter (Mel spectrogram is often better for speech)
        self.spectrogram = T.MelSpectrogram(sample_rate=16000, n_mels=64)
        
        # Target size for all images (spectrograms have fixed dimensions so we can batch them)
        self.target_size = (64, 64)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # 1. Get file path and label
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_name)
        
        # Extract label from filename (e.g., 'C03_5.wav' -> '5')
        label_str = file_name.split('_')[-1].split('.')[0]
        label = int(label_str)

        # 2. Load audio
        import librosa
        waveform, sample_rate = librosa.load(file_path, sr=16000)
        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0) # reshape to (1, time)
        
        # Convert to mono if it's stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 3. Audio Augmentations (Part B)
        if self.audio_aug:
            # Add a small amount of Gaussian noise
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise
            
            # Speed stretch +/- 3%
            speed_factor = random.choice([0.97, 1.03])
            # By resampling slightly, we change the speed (pitch will shift slightly too)
            new_sr = int(sample_rate * speed_factor)
            resampler = T.Resample(orig_freq=sample_rate, new_freq=new_sr)
            waveform = resampler(waveform)

        # 4. Convert Audio to Spectrogram Image
        spec = self.spectrogram(waveform)
        # Convert to log scale (decibels) to mimic human hearing, adding a small value to avoid log(0)
        spec = torch.log(spec + 1e-9)

        # 5. Resize to a strict fixed shape (64x64) so we can batch images properly
        # spec shape is (1, n_mels, time)
        spec = spec.unsqueeze(0)  # reshape to (1, 1, H, W) for interpolation
        spec = F.interpolate(spec, size=self.target_size, mode='bilinear', align_corners=False)
        spec = spec.squeeze(0)    # squeeze back to (1, H, W)

        # 6. Image Augmentations on Spectrograms (Part C)
        if self.image_aug:
            # Add image noise
            image_noise = torch.randn_like(spec) * 0.05
            spec = spec + image_noise
            
            # Horizontal Squeeze/Expand +/- 3% 
            # We scale the width, then interpolate the height back to 64
            scale_factor = random.choice([0.97, 1.03])
            new_width = int(self.target_size[1] * scale_factor)
            
            spec_aug = spec.unsqueeze(0)
            spec_aug = F.interpolate(spec_aug, size=(self.target_size[0], new_width), mode='bilinear', align_corners=False)
            # Re-interpolate back to strict fixed target size (64x64)
            spec_aug = F.interpolate(spec_aug, size=self.target_size, mode='bilinear', align_corners=False)
            spec = spec_aug.squeeze(0)

        # 7. Normalize the spectrogram (mean=0, std=1) to help the neural network learn faster
        mean = spec.mean()
        std = spec.std()
        spec = (spec - mean) / (std + 1e-9)

        return spec, label
