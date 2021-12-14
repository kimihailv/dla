import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, dilation):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        padding_size = int((kernel_size - 1) * dilation / 2)
        self.conv = weight_norm(nn.Conv1d(in_channels, out_channels,
                                          kernel_size=kernel_size,
                                          dilation=dilation,
                                          padding=padding_size))

    def forward(self, x):
        return x + self.conv(self.leaky_relu(x))


class DoubleResBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, dilations):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        padding_size = int((kernel_size - 1) * dilations[0] / 2)
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels,
                                          kernel_size=kernel_size,
                                          dilation=dilations[0],
                                          padding=padding_size))

        padding_size = int((kernel_size - 1) * dilations[1] / 2)

        self.conv2 = weight_norm(nn.Conv1d(in_channels, out_channels,
                                           kernel_size=kernel_size,
                                           dilation=dilations[1],
                                           padding=padding_size))

    def forward(self, x):
        x = self.conv1(self.leaky_relu(x))
        return x + self.conv2(self.leaky_relu(x))


class StackedResBlocks(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, dilations):
        super().__init__()
        self.layers = []

        for block_params in dilations:
            if len(block_params) == 1:
                self.layers.append(ResBlock(in_channels, out_channels,
                                            kernel_size, block_params[0]))

            else:
                self.layers.append(DoubleResBlock(in_channels, out_channels,
                                            kernel_size, block_params))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class MRF(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 resblocks_kernel_sizes,
                 resblocks_dilations):
        super().__init__()

        self.resblocks = nn.ModuleList([])

        for kernel_size, dilations in zip(resblocks_kernel_sizes, resblocks_dilations):
            self.resblocks.append(StackedResBlocks(in_channels, out_channels,
                                                   kernel_size, dilations))

        self.num_kernels = len(resblocks_kernel_sizes)

    def forward(self, x):
        out = 0

        for resblock in self.resblocks:
            out += resblock(x)

        return out / self.num_kernels


class Generator(nn.Module):
    def __init__(self,
                 in_channels,
                 initial_out_channels,
                 upsample_kernel_sizes,
                 resblocks_kernel_sizes,
                 resblocks_dilations):
        super().__init__()

        self.conv_init = weight_norm(nn.Conv1d(in_channels, initial_out_channels,
                                               7, 1, padding=3))
        self.upsample = []

        in_channels = initial_out_channels
        for layer_idx, kernel_size in enumerate(upsample_kernel_sizes, start=1):
            self.upsample.append(nn.LeakyReLU(0.1))

            out_channels = initial_out_channels // 2 ** layer_idx
            self.upsample.append(weight_norm(nn.ConvTranspose1d(in_channels, out_channels,
                                                                kernel_size=kernel_size,
                                                                stride=kernel_size // 2,
                                                                padding=(kernel_size - kernel_size // 2) // 2)))
            self.upsample.append(MRF(out_channels, out_channels,
                                     resblocks_kernel_sizes,
                                     resblocks_dilations))

            in_channels = out_channels

        self.upsample = nn.Sequential(*self.upsample)
        self.conv_last = weight_norm(nn.Conv1d(in_channels, 1, 7, 1, padding=3))

    def forward(self, x):
        x = self.conv_init(x)
        x = self.upsample(x)
        x = self.conv_last(F.leaky_relu(x))

        return torch.tanh(x)


class MSDBlock(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm = weight_norm if not use_spectral_norm else spectral_norm
        # convs parameters are from official implementation
        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.logits = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        x = self.logits(x)
        features.append(x)

        return x.flatten(1, -1), features


class MSDDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=4, stride=2, padding=2)
        self.discriminators = nn.ModuleList([
            MSDBlock(True),
            MSDBlock(),
            MSDBlock()
        ])

    def forward(self, x):
        logits_raw, features_raw = self.discriminators[0](x)
        x2 = self.pool(x)
        logits_x2, features_x2 = self.discriminators[1](x2)
        x4 = self.pool(x2)
        logits_x4, features_x4 = self.discriminators[2](x4)

        return (logits_raw, logits_x2, logits_x4), (features_raw, features_x2, features_x4)


class MPDBlock(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        # convs parameters are from official implementation
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))),
        ])
        self.logits = weight_norm(nn.Conv2d(1024, 1, (3, 1), stride=1, padding=(1, 0)))

    def forward(self, x):
        x = self.reshape(x)
        features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)

        x = self.logits(x)
        features.append(x)
        return x.flatten(1, -1), features

    def reshape(self, x):
        bs, c, t = x.shape

        reminder = t % self.period
        if reminder != 0:
            x = F.pad(x, (0, self.period - reminder), 'reflect')
            t = x.size(2)

        return x.view(bs, c, t // self.period, self.period)


class MPDDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([])

        for period in [2, 3, 5, 7, 11]:
            self.discriminators.append(MPDBlock(period))

    def forward(self, x):
        logits, features = [], []

        for discriminator in self.discriminators:
            l, f = discriminator(x)
            logits.append(l)
            features.append(f)

        return logits, features


class GeneratorV1(Generator):
    def __init__(self, in_channels):
        super().__init__(in_channels,
                         initial_out_channels=512,
                         upsample_kernel_sizes=[16, 16, 4, 4],
                         resblocks_kernel_sizes=[3, 7, 11],
                         resblocks_dilations=[[[1, 1], [3, 1], [5, 1]],
                                              [[1, 1], [3, 1], [5, 1]],
                                              [[1, 1], [3, 1], [5, 1]]])


class GeneratorV2(Generator):
    def __init__(self, in_channels):
        super().__init__(in_channels,
                         initial_out_channels=128,
                         upsample_kernel_sizes=[16, 16, 4, 4],
                         resblocks_kernel_sizes=[3, 7, 11],
                         resblocks_dilations=[[[1, 1], [3, 1], [5, 1]],
                                              [[1, 1], [3, 1], [5, 1]],
                                              [[1, 1], [3, 1], [5, 1]]])


class GeneratorV3(Generator):
    def __init__(self, in_channels):
        super().__init__(in_channels,
                         initial_out_channels=256,
                         upsample_kernel_sizes=[16, 16, 8],
                         resblocks_kernel_sizes=[3, 5, 7],
                         resblocks_dilations=[[[1], [2]],
                                              [[2], [6]],
                                              [[3], [12]]])
