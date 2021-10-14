from tqdm.notebook import tqdm


class Tokenizer:
    def __init__(self, data, filter_voc=True):
        self.idx2token = None
        self.token2idx = None
        self.voc = None
        self.filter_voc = filter_voc
        self._eps_token = '^'
        self.get_vocabulary(data)

    def get_vocabulary(self, data):
        voc = set()

        for item in tqdm(data):
            if self.filter_voc:
                voc.update(self.filter_text(item['text']))
            else:
                voc.update(item['text'])

        voc = sorted(list(voc))

        voc.append(self._eps_token)
        self.voc = set(voc)

        self.token2idx = {token: idx for idx, token in enumerate(voc)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

    def tokenize(self, text):
        if self.filter_voc:
            text = self.filter_text(text)
        return [self.token2idx[c] for c in text]

    def decode(self, tokenized):
        return ''.join(self.idx2token[t] for t in tokenized)

    @property
    def eps_token_id(self):
        return self.token2idx[self._eps_token]

    @property
    def eps_token(self):
        return self._eps_token

    @staticmethod
    def filter_text(text):
        return ''.join(filter(lambda c: c.isalpha() or c == ' ', text))

    def __len__(self):
        return len(self.voc)
