import torch.nn as nn
import torch
import librosa
from torchaudio import transforms
from dataclasses import dataclass


def init_weight(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        nn.init.normal_(m.weight.data, 0.0, 0.01)


def init_model(model):
    model.apply(init_weight)


def set_requires_grad(model, flag):
    for p in model.parameters():
        p.requires_grad = flag


def seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False


@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0


class MelSpectrogram(nn.Module):

    def __init__(self, config: MelSpectrogramConfig):
        super(MelSpectrogram, self).__init__()

        self.config = config

        self.mel_spectrogram = transforms.MelSpectrogram(
            sample_rate=config.sr,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            center=False,
        )

        # The is no way to set power in constructor in 0.5.0 version.
        self.mel_spectrogram.spectrogram.power = config.power

        # Default `torchaudio` mel basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))
        self.pad_size = (config.win_length - config.hop_length) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Expected shape is [B, T]
        :return: Shape is [B, n_mels, T']
        """
        x = nn.functional.pad(x, (self.pad_size, self.pad_size), 'reflect')
        mel = self.mel_spectrogram(x) \
            .clamp_(min=1e-5) \
            .log_()

        return mel

