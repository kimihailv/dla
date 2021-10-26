from itertools import groupby
from abc import abstractmethod
from ctcdecode import CTCBeamDecoder
from ..models import LM
from math import log
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
    def __init__(self, tokenizer, device, max_len=12, beam_size=4, lm_weight_path='',
                 alpha=0.5):
        super().__init__(tokenizer)
        self.max_len = max_len
        self.beam_size = beam_size
        self.lm = None

        if lm_weight_path != '':
            self.lm = LM(len(tokenizer)).to(device)
            self.lm.load_state_dict(torch.load(lm_weight_path))
            self.lm.eval()
            self.alpha = alpha

    def decode(self, model, batch, spec_lengths, return_best=True):
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
        beams = torch.cat([beams, new_tokens.unsqueeze(2)], dim=-1)
        states = [(prev_state, prev_context) for _ in range(self.beam_size)]
        vocab_size = len(self.tokenizer)

        for i in range(self.max_len):
            candidate_scores = torch.zeros(encoded.size(0), self.beam_size, vocab_size).to(encoded.device)

            for beam_idx, (prev_state, prev_context) in enumerate(states):
                logits, prev_state, prev_context = model.decoder_step(beams[:, beam_idx, -1],
                                                                      prev_context,
                                                                      prev_state,
                                                                      encoded)
                states[beam_idx] = (prev_state, prev_context)
                logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

                candidate_scores[:, beam_idx] = logprobs # bs x vocab_size
                candidate_scores[:, beam_idx] += scores[:, beam_idx].unsqueeze(1)

            candidate_scores = candidate_scores.reshape(encoded.size(0), -1) # bs x beam_size*vocab_size
            # bs x beam_size
            candidate_scores, flat_index = torch.topk(candidate_scores, self.beam_size, dim=1)
            beam_idx = flat_index // vocab_size
            new_token_idx = flat_index % vocab_size
            beams = torch.gather(beams, 1, beam_idx.unsqueeze(2).expand(-1, -1, beams.size(2)))
            beams = torch.cat([beams, new_token_idx.unsqueeze(2)], dim=2)

        beam_lens = (beams != self.tokenizer.eos_token_id).long().sum(dim=2).clamp(max=self.max_len - 1)
        mask = (beams != self.tokenizer.eos_token_id) & (beams != self.tokenizer.eps_token_id)

        if self.lm is not None:
            for batch_idx in range(beams.size(0)):
                asr_probs = scores[batch_idx] - beam_lens[batch_idx].log()
                lm_probs = self.lm.estimate_seq(beams[batch_idx], mask[batch_idx])
                scores[batch_idx] = asr_probs + self.alpha * lm_probs

        if return_best:
            beams = beams[torch.arange(beams.size(0)), scores.argmax(dim=1)]
            beam_lens = (beams != self.tokenizer.eos_token_id).long().sum(dim=1).clamp(max=self.max_len - 1).tolist()
            beams = beams.tolist()

            texts = []
            for token_ids, out_len in zip(beams, beam_lens):
                text = self.tokenizer.decode(token_ids[:out_len])
                texts.append(text)

            return texts

        sorted_ids = torch.argsort(scores, dim=1, descending=True)
        beams = torch.gather(beams, 1, sorted_ids.unsqueeze(2).expand(-1, -1, beams.size(2)))
        texts = []
        for batch_beams in beams:
            decoded_beams = []
            for beam in batch_beams:
                beam_len = (beam != self.tokenizer.eos_token_id).long().sum().clamp(max=self.max_len - 1).item()
                decoded_beams.append(self.tokenizer.decode(beam[:beam_len].tolist()))

            texts.append(decoded_beams)

        return texts
