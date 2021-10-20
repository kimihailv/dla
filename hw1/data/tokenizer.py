from tqdm.notebook import tqdm
from abc import abstractmethod
import youtokentome as yttm


class BaseTokenizer:
    def __init__(self, filter_voc=True):
        self.filter_voc = filter_voc
        self.voc = []
        self._eps_token = ''

    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def encode(self, text):
        pass

    @abstractmethod
    def decode(self, tokenized):
        pass

    @property
    def eps_token(self):
        return self._eps_token

    @property
    @abstractmethod
    def eps_token_id(self):
        pass

    @staticmethod
    def filter_text(text):
        return ''.join(filter(lambda c: c.isalpha() or c == ' ', text))


class Tokenizer(BaseTokenizer):
    def __init__(self, data, filter_voc=True):
        super().__init__(filter_voc)
        self.idx2token = None
        self.token2idx = None
        self._eps_token = '^'
        self.fit(data)

    def fit(self, data):
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

    def encode(self, text):
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


class BPETokenizer(BaseTokenizer):
    def __init__(self, data, filter_voc=True, vocab_size=1000):
        super(BPETokenizer, self).__init__(filter_voc)
        self.tokenizer = None
        self.vocab_size = vocab_size
        self._eps_token = '<PAD>'
        self.fit(data)

    def fit(self, data):
        with open('train_texts.txt', 'w+') as f:
            for item in tqdm(data):
                if self.filter_voc:
                    print(self.filter_text(item['text']), file=f)
                else:
                    print(item['text'], file=f)

        yttm.BPE.train(data='train_texts.txt', vocab_size=self.vocab_size,
                       model='bpe_model', pad_id=0)
        self.tokenizer = yttm.BPE(model='bpe_model')
        self.voc = self.tokenizer.vocab()

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, tokenized):
        return self.tokenizer.decode(tokenized)[0]

    @property
    def eps_token_id(self):
        return 0

    def __len__(self):
        return self.vocab_size
