import torch
import torch.nn as nn
import torch.nn.functional as F


class Listen(nn.Module):
    def __init__(self, n_layers, input_size, hidden_size, dropout=0):
        super().__init__()

        self.blstm = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size,
                             batch_first=True,
                             dropout=dropout,
                             bidirectional=True)
        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            self.layers.append(nn.LSTM(input_size=hidden_size * 4,
                                       hidden_size=hidden_size,
                                       batch_first=True,
                                       dropout=dropout,
                                       bidirectional=True))

    def forward(self, x):
        # x: N x L x F
        x, _ = self.blstm(x)
        # x: N x L x 2*hidden_size

        for layer in self.layers:
            if x.size(1) % 2 == 1:
                x = torch.cat((x, x[:, -1, :].unsqueeze(1)), dim=1)

            n, l, f = x.shape
            x = x.contiguous().view(n, l // 2, f * 2)
            # x: N x L x 4*hidden_size
            x, _ = layer(x)
            # x: N x L x 2*hidden_size

        return x


class Attend(nn.Module):
    def __init__(self, state_size, encoded_size, hidden_size, out_size):
        super().__init__()

        self.phi = torch.nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size)
        )

        self.psi = torch.nn.Sequential(
            nn.Linear(encoded_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size)
        )

    def forward(self, state, encoded):
        # state: N x state_size
        # encoded: N x L x encoded_size
        query = self.phi(state).unsqueeze(1)
        support = self.psi(encoded)
        # query: N x 1 x out_size
        # support: N x L x out_size
        scores = torch.bmm(support, query.transpose(2, 1).contiguous())
        # scores: N x L x 1
        attention_probs = F.softmax(scores, dim=1)
        context = (support * attention_probs).sum(dim=1)

        return context, attention_probs


class Spell(nn.Module):
    def __init__(self, emb_size, vocab_size, padding_idx,
                 context_size, hidden_size, dropout):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=emb_size,
                                      padding_idx=padding_idx)

        self.lstm = nn.LSTM(input_size=emb_size + hidden_size,
                            hidden_size=hidden_size,
                            num_layers=2,
                            batch_first=True,
                            dropout=dropout)

        self.predictor = nn.Sequential(
            nn.Linear(context_size + hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, vocab_size)
        )

    def forward(self, x, context, prev_state):
        # x: N
        # context: N x context_size
        # prev_state: (2 x N x hidden_size, 2 x N x hidden_size)
        x = self.embedding(x.unsqueeze(1))  # N x 1 x emb_size
        x = torch.cat((x, context.unsqueeze(1)), dim=2)

        state, new_prev_state = self.lstm(x, prev_state)
        logits = self.predictor(torch.cat((state.squeeze(), context), dim=1))
        return logits, new_prev_state


class LAS(nn.Module):
    def __init__(self,
                 vocab_size,
                 bos_idx,
                 padding_idx,
                 dropout=0,
                 encoder_n_layers=3,
                 hidden_size=256,
                 input_size=40,
                 emb_size=256,
                 context_size=256):
        super().__init__()
        self.bos_idx = bos_idx
        self.vocab_size = vocab_size
        self.encoder = Listen(n_layers=encoder_n_layers,
                              input_size=input_size,
                              hidden_size=hidden_size,
                              dropout=dropout)
        self.attention = Attend(state_size=hidden_size,
                                encoded_size=hidden_size * 2,
                                hidden_size=hidden_size * 3,
                                out_size=context_size)
        self.decoder = Spell(emb_size=emb_size,
                             vocab_size=vocab_size,
                             padding_idx=padding_idx,
                             context_size=context_size,
                             hidden_size=hidden_size,
                             dropout=dropout)

        self.dec_start_h = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.dec_start_c = nn.Linear(hidden_size * 2, hidden_size * 2)

    def forward(self, x, mode='train', sampling_from_prev_rate=0.1):
        x['specs'] = x['specs'].transpose(2, 1)
        batch_size = x['specs'].size(0)
        encoded = self.encoder(x['specs'])
        batch_range = torch.arange(batch_size).to(x['specs'].device)
        last_hidden = encoded[batch_range, x['specs_len']]

        prev_h = self.dec_start_h(last_hidden)
        prev_c = self.dec_start_c(last_hidden)
        # prev_h: N x hidden_size * 2
        # prev_c: N x hidden_size * 2

        prev_h = prev_h.view(batch_size, last_hidden.size(1) // 2, 2).permute(2, 0, 1).contiguous()
        prev_c = prev_c.view(batch_size, last_hidden.size(1) // 2, 2).permute(2, 0, 1).contiguous()
        prev_state = (prev_h, prev_c)
        context, attention_probs = self.attention(prev_h[0], encoded)

        bos_logits = torch.full((batch_size,), self.bos_idx, dtype=torch.int64)
        bos_logits = torch.log(F.one_hot(bos_logits, num_classes=self.vocab_size) + 1e-9).to(x['specs'].device)
        seq_logits = [bos_logits.unsqueeze(1)]

        for step_idx in range(x['targets'].shape[1] - 1):
            if mode == 'train':
                next_token = self.get_next_token_train(x,
                                                       step_idx,
                                                       seq_logits[-1],
                                                       sampling_from_prev_rate)
            else:
                next_token = self.get_next_token_eval(x,
                                                      step_idx,
                                                      seq_logits[-1],
                                                      sampling_from_prev_rate)

            logits, prev_state = self.decoder(next_token,
                                              context,
                                              prev_state)
            # 2 x N x hidden_size
            prev_h = prev_state[0][1]
            context, attention_probs = self.attention(prev_h, encoded)
            seq_logits.append(logits.unsqueeze(1))

        return torch.cat(seq_logits, dim=1)

    @staticmethod
    def get_next_token_train(x, step_idx, prev_logits, sampling_from_prev_rate):
        if sampling_from_prev_rate == 0 or step_idx == 0:
            return x['targets'][:, step_idx]

        to_sample = torch.bernoulli(torch.FloatTensor([sampling_from_prev_rate])).item()

        if to_sample == 1:
            return torch.multinomial(F.softmax(prev_logits.squeeze(), dim=1), num_samples=1).squeeze()

        return x['targets'][:, step_idx]

    @staticmethod
    def get_next_token_eval(x, step_idx, prev_logits, sampling_from_prev_rate):
        return prev_logits.squeeze().argmax(dim=-1)

    def calc_loss(self, x, device, loss, mode, return_output=False):
        x['specs'] = x['specs'].to(device)
        x['specs_len'] = torch.LongTensor(x['specs_len']).to(device)
        x['targets'] = x['targets'].to(device)
        logits = self.forward(x, mode)
        loss_v = loss(logits.transpose(2, 1).contiguous(), x['targets'])

        if return_output:
            return loss_v, F.log_softmax(logits, dim=-1)

        return loss_v
