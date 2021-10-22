from tqdm.notebook import tqdm
from abc import abstractmethod
from random import randint
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

    @abstractmethod
    def dump(self):
        pass

    @abstractmethod
    def load(self, state):
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
        self.voc = set(voc)  # to remove

        self.token2idx = {token: idx for idx, token in enumerate(voc)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        # fix order
        self.voc = [self.idx2token[idx] for idx in range(len(voc))]

    def encode(self, text):
        if self.filter_voc:
            text = self.filter_text(text)
        return [self.token2idx[c] for c in text]

    def decode(self, tokenized):
        return ''.join(self.idx2token[t] for t in tokenized)

    def dump(self):
        return self

    def load(self, state):
        self.voc = state.voc
        self.filter_voc = state.filter_voc
        self._eps_token = state.eps_token
        self.token2idx = state.token2idx
        self.idx2token = state.idx2token

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
    def __init__(self, data, save_dir, filter_voc=True, vocab_size=1000,
                 use_bos=False, use_eos=False):
        super(BPETokenizer, self).__init__(filter_voc)
        self.save_dir = save_dir
        self.tokenizer = None
        self.vocab_size = vocab_size
        self._eps_token = '<PAD>'
        self.use_bos = use_bos
        self.use_eos = use_eos
        self.fit(data)

    def fit(self, data):
        suffix = randint(0, 100000)
        with open(f'{self.save_dir}/train_texts_{suffix}.txt', 'w+') as f:
            for item in tqdm(data):
                if self.filter_voc:
                    print(self.filter_text(item['text']), file=f)
                else:
                    print(item['text'], file=f)

        yttm.BPE.train(data=f'{self.save_dir}/train_texts_{suffix}.txt', vocab_size=self.vocab_size,
                       model=f'{self.save_dir}/bpe_model_{suffix}', pad_id=0, unk_id=1, bos_id=2, eos_id=3)

        self.tokenizer = yttm.BPE(model=f'{self.save_dir}/bpe_model_{suffix}')
        self.voc = self.tokenizer.vocab()

    def encode(self, text):
        return self.tokenizer.encode(text, bos=self.use_bos, eos=self.use_eos)

    def decode(self, tokenized):
        return self.tokenizer.decode(tokenized)[0]

    @property
    def eps_token_id(self):
        return self.tokenizer.subword_to_id(self._eps_token)

    @property
    def bos_token_id(self):
        return self.tokenizer.subword_to_id('<BOS>')

    def __len__(self):
        return self.vocab_size

    def dump(self):
        state = {'save_dir': self.save_dir,
                 'voc': self.voc,
                 'filter_voc': self.filter_voc,
                 'eps_token': self._eps_token,
                 'vocab_size': self.vocab_size,
                 'use_bos': self.use_bos,
                 'use_eos': self.use_eos
                 }
        return state

    def load(self, state):
        self.voc = state['voc']
        self.filter_voc = state['filter_voc']
        self._eps_token = state['eps_token']
        self.tokenizer = yttm.BPE(model=state['save_dir'])
        self.vocab_size = state['vocab_size']
        self.use_bos = state['use_bos']
        self.use_eos = state['use_eos']
