import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_size,
                 hidden_size,
                 num_layers,
                 voc_size):
        super().__init__()
        self.prenet = torch.nn.Sequential(
            nn.Conv2d(1, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 1)
        )
        self.encoder = nn.LSTM(input_size=input_size * 16, hidden_size=hidden_size,
                               num_layers=num_layers, bidirectional=True, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, voc_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        # x: N x 1 x L x F
        x = self.prenet(x)
        # x: N x C x L x F
        bs, c, l, f = x.size()
        x = x.transpose(3, 2).reshape(bs, c * f, l)
        encoded, _ = self.encoder(x.transpose(2, 1))
        # encoded: N x L x D
        logprobs = self.head(encoded)
        # logprobs: N x L x V
        return logprobs

    def calc_loss(self, batch, device, loss, return_output=False):
        logprobs = self(batch['specs'].to(device))
        target = batch['targets'].to(device)
        input_len = batch['specs_len']
        target_len = batch['targets_len']

        if return_output:
            return loss(logprobs.permute(1, 0, 2), target, input_len, target_len), logprobs

        return loss(logprobs.permute(1, 0, 2), target, input_len, target_len)
