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
        beams = bos_logits.argmax(dim=1).unsqueeze(1).unsqueeze(2).expand(-1, self.beam_size, -1)

        logits, prev_state, prev_context = model.decoder_step(bos_logits.argmax(dim=1),
                                                              prev_context,
                                                              prev_state,
                                                              encoded)
        logprobs = torch.nn.functional.log_softmax(logits, dim=1)
        # beams, scores: bs x beam_size
        scores, new_tokens = torch.topk(logprobs, self.beam_size, dim=-1)
        # beams: bs x beam_size x 2
        beams = torch.cat([beams, new_tokens], dim=-1)
        states = [(prev_state, prev_context) for _ in range(self.beam_size)]
        vocab_size = len(self.tokenizer)

        for i in range(self.max_len):
            candidate_scores = torch.zeros(encoded.size(0), self.beam_size, vocab_size)

            for beam_idx, (prev_state, prev_context) in enumerate(states):
                logits, prev_state, prev_context = model.decoder_step(beams[:, beam_idx, -1],
                                                                      prev_context,
                                                                      prev_state,
                                                                      encoded)
                states[i] = (prev_state, prev_context)
                candidate_scores[:, beam_idx] = torch.nn.functional.log_softmax(logits, dim=-1)  # bs x vocab_size
                candidate_scores[:, beam_idx] += scores[:, beam_idx].unsqueeze(1)

            candidate_scores = candidate_scores.reshape(encoded.size(0), -1) # bs x beam_size*vocab_size
            # bs x beam_size
            candidate_scores, flat_index = torch.topk(candidate_scores, self.beam_size, dim=1)
            beam_idx = flat_index // vocab_size
            new_token_idx = flat_index % vocab_size

            beams = torch.gather(beams, 1, beam_idx.unsqueeze(2))
            beams = torch.cat([beams, new_token_idx.unsqueeze(2)], dim=2)
            scores = candidate_scores

        top_beams = beams[torch.arange(beams.size(0)), scores.argmax(dim=1)]
        beam_lens = (top_beams != self.tokenizer.eps_token_id).long().sum(dim=1).clamp(max=self.max_len - 1).tolist()
        top_beams = top_beams.tolist()

        texts = []
        for token_ids, out_len in zip(top_beams, beam_lens):
            text = self.tokenizer.decode(token_ids[:out_len])
            texts.append(text)

        return texts
