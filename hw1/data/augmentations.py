import librosa.effects as aug
import numpy as np
import torch
from torchaudio.transforms import MelSpectrogram


class AddNoise:
    def __init__(self, std=0.001):
        self.std = std

    def __call__(self, wav):
        return wav + np.random.normal(loc=0, scale=self.std, size=wav.shape)


class TimeStretch:
    def __init__(self, low_rate=0.95, high_rate=1.5):
        self.low_rate = low_rate
        self.high_rate = high_rate

    def __call__(self, wav):
        rate = np.random.uniform(self.low_rate, self.high_rate)
        return aug.time_stretch(wav, rate)


class PitchShift:
    def __init__(self, sr, low_steps=-2, high_steps=3):
        self.sr = sr
        self.low_steps = low_steps
        self.high_steps = high_steps

    def __call__(self, wav):
        steps = np.random.randint(self.low_steps, self.high_steps)
        return aug.pitch_shift(wav, self.sr, n_steps=steps)


class MelTransform:
    def __init__(self, transform=None, **mel_spectogram_kwargs):
        self.mel_spectogram = MelSpectrogram(**mel_spectogram_kwargs)
        self.transform = transform

    def __call__(self, wav):
        mels = self.mel_spectogram(torch.from_numpy(wav).float())

        if self.transform is not None:
            mels = self.transform(mels)

        return mels


class Compose:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data


class RandomApply:
    def __init__(self, transform, p=0.5):
        self.transform = transform
        self.p = p

    def __call__(self, data):
        if np.random.binomial(1, self.p) == 1:
            return self.transform(data)

        return data

