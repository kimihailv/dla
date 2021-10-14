from itertools import groupby
import torch


class TextDecoder:
    def __init__(self, tokenizer, pick_by='argmax'):
        self.tokenizer = tokenizer
        self.pick_by = pick_by

    def ctc_decode(self, text):
        parts = text.split(self.tokenizer.eps_token)
        decoded = []

        for part in parts:
            decoded += [c for c, _ in groupby(part)]

        return ''.join(decoded)

    def decode(self, probs, lengths):
        batch_token_ids = self.pick_tokens(probs)
        texts = []
        for token_ids, spec_len in zip(batch_token_ids, lengths):
            text = self.tokenizer.decode(token_ids[:spec_len])
            text = self.ctc_decode(text)
            texts.append(text)

        return texts

    def pick_tokens(self, probs):
        if self.pick_by == 'argmax':
            return torch.argmax(probs, dim=-1).tolist()