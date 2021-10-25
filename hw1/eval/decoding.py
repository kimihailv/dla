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


class Seq2SeqBeamSearchDecoder(BaseTextDecoder):
    def __init__(self, tokenizer, max_len=12, beam_size=4):
        super().__init__(tokenizer)
        self.max_len = max_len
        self.beam_size = beam_size

    def decode(self, model, batch, spec_lengths):
        bos_logits, prev_context, prev_state, encoded, attention_probs = model.start_decode(batch)
        bos_token_id = bos_logits.argmax(dim=1)

        batch_beams = [[([token], 0)] for token in bos_token_id]

        for i, beams in enumerate(batch_beams):
            batch_beams[i] = self.beam_search(model, prev_context, prev_state, encoded, beams)

        return

    def beam_search(self, model, prev_context, prev_state, encoded, beams):
        tokens_ids = torch.arange(len(self.tokenizer)).to(model.device)
        for i in range(self.max_len):
            candidates = []
            for seq, score in beams:
                if seq[-1].item() != self.tokenizer.eos_token_id:
                    logits, prev_state, prev_context = model.decoder_step(seq[-1].unsqueeze(0),
                                                                          prev_context,
                                                                          prev_state,
                                                                          encoded)

                    logprobs = torch.nn.functional.log_softmax(logits, dim=-1).tolist()

                    for token_id, logprob in zip(tokens_ids, logprobs):
                        candidates.append((seq + [token_id], score + logprob))
                else:
                    candidates.append((seq, score))

            beams = sorted(candidates, key=lambda c: -c[1])[:self.beam_size]

        results = []

        for seq, score in beams:
            results.append(torch.hstack(seq))

        return torch.nn.utils.rnn.pad_sequence(results, padding_value=self.tokenizer.eps_token_id)