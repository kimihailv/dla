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
            nn.Conv1d(hidden_size, hidden_size * 2, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_size * 2, voc_size, 1),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        # N x L x C
        encoded, _ = self.encoder(x.transpose(2, 1))
        # N x C x L
        logprobs = self.head(encoded.transpose(2, 1))

        return logprobs.permute(2, 0, 1)

    def calc_loss(self, batch, device, loss, return_output=False):
        logprobs = self.model(batch['mels'].to(device))
        target = batch['target_tokens_idx'].to(device)
        input_len = batch['mel_len'].to(device)
        target_len = batch['target_len'].to(device)

        if return_output:
            return loss(logprobs, target, input_len, target_len), logprobs

        return loss(logprobs, target, input_len, target_len)
