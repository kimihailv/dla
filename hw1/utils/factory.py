from ..data.augmentations import *
from ..data.tokenizer import Tokenizer, BPETokenizer
from ..data.datasets import *
from ..models import *
from ..logging import WandbLogger
from torchaudio.transforms import FrequencyMasking, TimeMasking
from torch.utils.data import Subset
from ..eval.decoding import *
import torch
import math


def CosineWithWarmup(
        optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


inventory = {
    'model': {
        'QuartzNet': QuartzNet,
        'LAS': LAS
    },

    'optimizer': {
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW,
        'SGD': torch.optim.SGD
    },

    'scheduler': {
        'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR,
        'OneCycleLR': torch.optim.lr_scheduler.OneCycleLR,
        'CosineWithWarmup': CosineWithWarmup
    },

    'loss': {
        'CTCLoss': torch.nn.CTCLoss,
        'CrossEntropyLoss': torch.nn.CrossEntropyLoss
    },

    'dataset': {
        'Librispeech': LibrispeechDataset,
        'LJ': LJDataset,
        'HDF5Dataset': HDF5Dataset,
        'TestDataset': TestDataset
    },

    'aug': {
        'AddNoise': AddNoise,
        'PitchShift': PitchShift,
        'TimeStretch': TimeStretch,
        'MelTransform': MelTransform,
        'Compose': Compose,
        'RandomApply': RandomApply,
        'FrequencyMasking': FrequencyMasking,
        'TimeMasking': TimeMasking
    },

    'tokenizer': {
        'Tokenizer': Tokenizer,
        'BPETokenizer': BPETokenizer
    },

    'text_decoder': {
        'CTCGreedyDecoder': CTCGreedyDecoder,
        'BeamSearchDecoder': BeamSearchDecoder,
        'GreedyDecoder': GreedyDecoder,
        'Seq2SeqBeamSearchDecoder': Seq2SeqBeamSearchDecoder
    },

    'logger': {
        'WandbLogger': WandbLogger
    }
}


def make_generic(section, params):
    return inventory[section][params['constructor']](**params['args'])


def make_aug(params):
    augmentation = make_generic('aug', params)

    if params['apply_prob'] == 1:
        return augmentation

    return RandomApply(augmentation, p=params['apply_prob'])


def make_mel_transform(params):
    if 'transform' in params:
        transform = [make_aug(p) for p in params['transform']]
        transform = Compose(*transform)
    else:
        transform = None

    return MelTransform(transform=transform, **params["args"])


def make_dataset(dataset_params, common_params, tokenizer=None):
    cls = inventory['dataset'][common_params['constructor']]
    if isinstance(dataset_params['split'], str):
        dataset = cls(tokenizer=tokenizer,
                      root=common_params['root_dir'],
                      url=dataset_params['split'],
                      download=True)
    else:
        dataset = cls(tokenizer=tokenizer,
                      root=common_params['root_dir'],
                      download=True)

    if tokenizer is None:
        common_params['tokenizer']['args']['data'] = dataset
        tokenizer = make_generic('tokenizer', common_params['tokenizer'])

    dataset.tokenizer = tokenizer
    dataset = filter_dataset(dataset, common_params['max_duration'],
                             common_params['max_target_len'])

    if not isinstance(dataset_params['split'], str):
        start_frac, end_frac = dataset_params['split']
        total_len = len(dataset)
        ids = list(range(int(start_frac * total_len), int(end_frac * total_len)))
        dataset = Subset(dataset, ids)

    return dataset, tokenizer
