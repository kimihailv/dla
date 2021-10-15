import torch
from torch.utils.data import Dataset
from functools import partial
from librosa import load
import soundfile as sf
import numpy as np


def get_preprocess_fn(tokenizer, sound_dir='', sound_ext='flac', sr=22050):
    def preprocess_dataset(sample, tokenizer, sound_dir, sound_ext, sr):

        if sound_dir != '':
            filename = sample['file'].split('/')[-1]
            sample['file'] = f'{sound_dir}/{filename}'

        if sound_ext == 'flac':
            sample['wav'] = sf.read(sample['file'])[0]
        else:
            sample['wav'] = load(sample['file'], sr=sr)
        sample['target_tokens_idx'] = tokenizer.tokenize(sample['text'])

        return sample

    return partial(preprocess_dataset,
                   tokenizer=tokenizer,
                   sound_dir=sound_dir,
                   sound_ext=sound_ext,
                   sr=sr)


def get_filter_fn(max_duration, max_target_len, sr=22050):
    def filter_by_len(sample, max_duration, max_target_len, sr):
        if len(sample['wav']) / sr > max_duration:
            return False

        if len(sample['text']) > max_target_len:
            return False

        return True

    return partial(filter_by_len,
                   max_duration=max_duration,
                   max_target_len=max_target_len,
                   sr=sr)


class BaseDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, idx):
        item = self.data[idx]
        item['wav'] = np.array(item['wav'])
        item['target_tokens_idx'] = np.array(item['target_tokens_idx'])

        return item

    def __len__(self):
        return 20
        # return len(self.data)


class Collator:
    def __init__(self, wav_transform=None, mel_transform=None):
        self.wav_transform = wav_transform
        self.mel_transform = mel_transform

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
            specs.append(spec.transpose(1, 0))
            specs_len.append(spec.shape[1])
            targets.append(torch.from_numpy(item['target_tokens_idx']))
            targets_len.append(len(item['target_tokens_idx']))

        batch = {
            'wavs': wavs,
            'targets': torch.nn.utils.rnn.pad_sequence(targets, batch_first=True),
            'targets_len': targets_len,
            'specs': torch.nn.utils.rnn.pad_sequence(specs, batch_first=True),
            'specs_len': specs_len
        }

        for k in samples[0].keys():
            if k in ['wav', 'target_tokens_idx']:
                continue
            batch[k] = []

            for sample in samples:
                batch[k].append(sample[k])

        return batch
