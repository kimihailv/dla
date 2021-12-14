import torch.nn as nn
from torchaudio import transforms


def init_model(model):
    def init_weight(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

    model.apply(init_weight)


def set_requires_grad(model, flag):
    for p in model.parameters():
        p.requires_grad = flag


class Featurizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.featurizer = transforms.MelSpectrogram(
            sample_rate=22_050,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=80,
            center=False
        )

    def forward(self, x):
        spec = self.featurizer(x)
        return spec.clamp(min=1e-5).log()
