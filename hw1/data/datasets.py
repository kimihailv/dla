import torch
from torch.utils.data import Subset
from torchaudio.datasets import LJSPEECH, LIBRISPEECH


def filter_dataset(dataset, max_duration, max_target_len):
    exclude_ids = set()
    for item_idx, item in enumerate(dataset):
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
            if k in ['wav', 'target_tokens_idx']:
                continue
            batch[k] = []

            for sample in samples:
                batch[k].append(sample[k])

        return batch
