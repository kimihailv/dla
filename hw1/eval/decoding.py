from itertools import groupby
from abc import abstractmethod
from ctcdecode import CTCBeamDecoder
import torch


class BaseTextDecoder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def decode(self, model, probs, spec_lengths):
        pass


class GreedyDecoder(BaseTextDecoder):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    def decode(self, model, probs, spec_lengths):
        batch_token_ids = torch.argmax(probs, dim=-1).tolist()
        texts = []
        for token_ids in batch_token_ids:
            text = self.tokenizer.decode(token_ids)
            texts.append(text)

        return texts


class CTCGreedyDecoder(BaseTextDecoder):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    def ctc_decode(self, text):
        parts = text.split(self.tokenizer.eps_token)
        decoded = []

        for part in parts:
            decoded += [c for c, _ in groupby(part)]

        return ''.join(decoded)

    def decode(self, model, probs, spec_lengths):
        batch_token_ids = torch.argmax(probs, dim=-1).tolist()
        texts = []
        for token_ids, spec_len in zip(batch_token_ids, spec_lengths):
            text = self.tokenizer.decode(token_ids[:spec_len])
            text = self.ctc_decode(text)
            texts.append(text)

        return texts


class BeamSearchDecoder(CTCGreedyDecoder):
    def __init__(self, tokenizer, model_path=None, beam_width=100, alpha=0, beta=0):
        super().__init__(tokenizer)

        self.decoder = CTCBeamDecoder(
            labels=self.tokenizer.voc,
            model_path=model_path,
            alpha=alpha,
            beta=beta,
            cutoff_top_n=len(self.tokenizer),
            cutoff_prob=1.0,
            beam_width=beam_width,
            num_processes=2,
            blank_id=self.tokenizer.eps_token_id,
            log_probs_input=True
        )

    def decode(self, model, probs, spec_lengths):
        beams, _, _, out_lens = self.decoder.decode(probs)
        beams = beams[:, 0, :].tolist()
        out_lens = out_lens[:, 0].tolist()
        texts = []
        for token_ids, out_len in zip(beams, out_lens):
            text = self.tokenizer.decode(token_ids[:out_len])
            text = self.ctc_decode(text)
            texts.append(text)

        return texts
