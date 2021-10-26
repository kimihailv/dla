import torch


class LM(torch.nn.Module):
    def __init__(self, n_tokens):
        super().__init__()
        self.emb = torch.nn.Embedding(num_embeddings=n_tokens, embedding_dim=256)
        self.gru = torch.nn.GRU(256, 512, num_layers=2, batch_first=True)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, n_tokens)
        )

    def forward(self, x):
        x = self.emb(x)
        hiddens, _ = self.gru(x)
        return self.head(hiddens)

    def get_prob_of_tokens(self, prev_tokens, tokens=None):
        logits = self(prev_tokens)
        logprobs = torch.nn.functional.log_softmax(logits, dim=2)[:, -1]

        if tokens is None:
            return logprobs

        return torch.gather(logprobs, 1, tokens)

    def estimate_seq(self, seq, mask):
        # seq: bs x L
        log_probs = torch.zeros((seq.size(0),)).to(seq.device)

        for i in range(1, seq.size(1)):
            logits = self(seq[:, :i])
            next_token_dist = torch.nn.functional.log_softmax(logits, dim=2)[:, -1]  # bs x 1 x vocab
            next_token_probs = next_token_dist[torch.arange(seq.size(0)), seq[:, i]]
            log_probs += next_token_probs * mask[:, i]

        return log_probs