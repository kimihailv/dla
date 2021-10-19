import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride, dilation=dilation)

        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class SubBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.sep_conv = nn.Conv1d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels)
        self.poinwise_conv = nn.Conv1d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.poinwise_conv(self.sep_conv(x))
        return F.relu(self.bn(x))


class Block(nn.Module):
    def __init__(self, n_repeat, in_channels, out_channels, kernel_size):
        super().__init__()
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )
        pad = kernel_size // 2
        self.subblocks = [SubBlock(in_channels, out_channels, kernel_size, pad)]

        for _ in range(n_repeat - 1):
            self.subblocks.append(SubBlock(out_channels, out_channels, kernel_size, pad))

        self.subblocks = nn.Sequential(*self.subblocks)

    def forward(self, x):
        identity = x.clone()
        if self.shortcut is not None:
            identity = self.shortcut(identity)

        return F.relu(self.subblocks(x) + identity)


class QuartzNet(nn.Module):
    num_channels = [256, 256, 512, 512, 512]
    kernel_sizes = [33, 39, 51, 63, 75]

    def __init__(self, in_channels, n_blocks, n_subblocks, voc_size):
        super().__init__()
        self.conv1 = ConvBnReLU(in_channels, 256, 33, stride=2)

        self.blocks = []
        n_repeat_block = n_blocks // 5
        in_channels = 256

        for out_channels, kernel_size in zip(self.num_channels, self.kernel_sizes):
            for _ in range(n_repeat_block):
                self.blocks.append(Block(n_subblocks, in_channels, out_channels, kernel_size))
                in_channels = out_channels

        self.blocks = nn.Sequential(*self.blocks)

        self.final = nn.Sequential(
            SubBlock(512, 512, 87, 0),
            ConvBnReLU(512, 1024, 1),
            nn.Conv1d(1024, voc_size, kernel_size=1, dilation=2),
            nn.LogSoftmax(1)
        )

    def forward(self, x):
        x = self.conv1(x)
        # x: N x C x L
        x = self.blocks(x)
        x = self.final(x)

        return x.transpose(2, 1)
