import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_size,
                 hidden_size,
                 num_layers,
                 voc_size):
        super().__init__()
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, voc_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        # x: N x L x C
        encoded, _ = self.encoder(x)
        # encoded: N x L x D
        logprobs = self.head(encoded)
        # logprobs: N x L x V
        return logprobs

    def calc_loss(self, batch, device, loss, return_output=False):
        logprobs = self(batch['specs'].to(device))
        target = batch['targets'].to(device)
        input_len = batch['specs_len'].tolist()
        target_len = batch['targets_len'].tolist()

        if return_output:
            return loss(logprobs.permute(1, 0, 2), target, input_len, target_len), logprobs

        return loss(logprobs.permute(1, 0, 2), target, input_len, target_len)
