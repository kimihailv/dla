from ..data.augmentations import *
from ..data.tokenizer import Tokenizer
from ..data.datasets import get_preprocess_fn, get_filter_fn, BaseDataset
from ..models import *
from ..logging import WandbLogger
from torchaudio.transforms import FrequencyMasking, TimeMasking
import datasets
import torch

datasets.set_caching_enabled(False)
inventory = {
    'model': {
        'LSTM': LSTM,
        'QuartzNet': QuartzNet
    },

    'optimizer': {
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW,
        'SGD': torch.optim.SGD
    },

    'scheduler': {
        'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR,
        'OneCycleLR': torch.optim.lr_scheduler.OneCycleLR
    },

    'loss': {
        'CTCLoss': torch.nn.CTCLoss,
        'CrossEntropyLoss': torch.nn.CrossEntropyLoss
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


def make_dataset(dataset_params, preprocess_params, tokenizer=None):
    if dataset_params['type'] == 'hugging_face':
        if dataset_params['load_from'] == 'disk':
            dataset_path = dataset_params['root_dir'] + '/' + dataset_params['name']
            dataset = datasets.load_from_disk(dataset_path)
        else:
            if 'part' in dataset_params:
                dataset = datasets.load_dataset(dataset_params['name'], dataset_params['part'],
                                                split=dataset_params['split'])
            else:
                dataset = datasets.load_dataset(dataset_params['name'])
    else:
        dataset = None

    if tokenizer is None:
        tokenizer = Tokenizer(dataset)

    if dataset_params['sound_dir'] != '':
        dataset_params['sound_dir'] = dataset_params['root_dir'] + '/' + dataset_params['sound_dir']

    dataset = dataset.map(get_preprocess_fn(tokenizer,
                                            dataset_params['sound_dir'],
                                            dataset_params['sound_ext'],
                                            preprocess_params['sr']),
                          num_proc=2,
                          remove_columns=preprocess_params['remove_columns'])
    print('Preprocessing finished')
    dataset = dataset.filter(get_filter_fn(preprocess_params['max_duration'],
                                           preprocess_params['max_target_len'],
                                           preprocess_params['sr']),
                             num_proc=1)
    print('Filtering finished')
    return BaseDataset(dataset), tokenizer
