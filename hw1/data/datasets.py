import h5py
import torch
from torch.utils.data import Subset
from torch.utils.data.dataset import T_co
from torchaudio.datasets import LJSPEECH, LIBRISPEECH
from tqdm import tqdm


def filter_dataset(dataset, max_duration, max_target_len):
    exclude_ids = set()
    for item_idx, item in tqdm(enumerate(dataset), total=len(dataset), leave=True, position=0):
        if len(item['wav']) / dataset.sample_rate > max_duration or\
                len(item['text']) > max_target_len:
            exclude_ids.add(item_idx)

    selected_ids = set(range(len(dataset))) - exclude_ids

    return Subset(dataset, sorted(list(selected_ids)))


class LJDataset(LJSPEECH):
    def __init__(self, tokenizer=None, **dataset_kwargs):
        super().__init__(**dataset_kwargs)
        self.tokenizer = tokenizer
        self.sample_rate = 22050

    def __getitem__(self, idx):
        item = super().__getitem__(idx)

        if self.tokenizer is not None:
            data = {
                'wav': item[0][0].numpy(),
                'target_tokens_idx': self.tokenizer.encode(item[3]),
                'text': self.tokenizer.filter_text(item[3])
            }
        else:
            data = {
                'wav': item[0][0].numpy(),
                'text': item[3]
            }

        return data


class LibrispeechDataset(LIBRISPEECH):
    def __init__(self, tokenizer, **dataset_kwargs):
        super().__init__(**dataset_kwargs)
        self.tokenizer = tokenizer
        self.sample_rate = 16000

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        data = {
            'wav': item[0][0].numpy(),
            'target_tokens_idx': self.tokenizer(item[2]),
            'text': self.tokenizer.filter_text(item[2])
        }

        return data


class Collator:
    def __init__(self, wav_transform=None, mel_transform=None, input_len_div_factor=1):
        self.wav_transform = wav_transform
        self.mel_transform = mel_transform
        self.div_factor = input_len_div_factor

    def __call__(self, samples):
        specs = []
        targets = []
        wavs = []
        specs_len = []
        targets_len = []

        for item in samples:
            wav = item['wav']
            if self.wav_transform is not None:
                wav = self.wav_transform(wav)

            spec = self.mel_transform(wav)

            wavs.append(wav)
            specs.append(spec.clamp(min=1e-5).log().transpose(1, 0))
            specs_len.append(spec.shape[1] // self.div_factor)
            targets.append(torch.IntTensor(item['target_tokens_idx']))
            targets_len.append(len(item['target_tokens_idx']))

        batch = {
            'wavs': wavs,
            'targets': torch.nn.utils.rnn.pad_sequence(targets, batch_first=True),
            'targets_len': targets_len,
            'specs': torch.nn.utils.rnn.pad_sequence(specs, batch_first=True).transpose(2, 1),
            'specs_len': specs_len
        }

        for k in samples[0].keys():
            if k in ['wav', 'targets']:
                continue
            batch[k] = []

            for sample in samples:
                batch[k].append(sample[k])

        return batch


class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, root_dir, url=None):
        self.tokenizer = tokenizer
        if url is None:
            self.data = h5py.File(root_dir, 'r')
        else:
            self.data = h5py.File(root_dir, 'r')['url']

        self.keys = list(self.data.keys())

    def __getitem__(self, idx):
        item = self.data[self.keys[idx]]
        text = item['text'].asstr()[()]
        item = {'wav': item['wav'],
                'target_tokens_idx': self.tokenizer(text),
                'text': self.tokenizer.filter_text(text)
                }
        return item
