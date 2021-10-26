import torch
import gdown
from os.path import exists
from argparse import ArgumentParser
from .factory import make_generic, make_mel_transform
from ..data.datasets import Collator
from ..eval.metrics import *
from json import load, dump
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('-c',
                        action='store',
                        type=str,
                        help='path to config file')

    parser.add_argument('-t',
                        action='store',
                        type=str,
                        help='path to dir with test data')

    parser.add_argument('-o',
                        action='store',
                        type=str,
                        help='path to output file')

    return parser.parse_args()


def download_checkpoint(config):
    if exists(config['checkpoint_path']):
        return

    gdown.download(config['checkpoint_link'], config['checkpoint_path'])
    if 'bpe_link' in config:
        gdown.download(config['bpe_link'], config['bpe_path'])


def make_test_loader(config, tokenizer_state, test_data_dir):
    tokenizer = make_generic('tokenizer', config['tokenizer_params'])
    tokenizer.load(tokenizer_state)
    config['dataset_params']['args']['tokenizer'] = tokenizer

    if config['dataset_params']['constructor'] == 'TestDataset':
        config['dataset_params']['args']['data_dir'] = test_data_dir

    dataset = make_generic('dataset', config['dataset_params'])
    mel_transform = make_mel_transform(config['dataset_params']['mel_transform'])
    collator = Collator(wav_transform=None,
                        mel_transform=mel_transform,
                        input_len_div_factor=config['dataset_params']['input_len_div_factor'])

    return DataLoader(dataset, batch_size=config['dataset_params']['batch_size'],
                      collate_fn=collator), tokenizer


@torch.no_grad()
def eval_model(out_file, model, device, test_loader, greedy_decoder, beam_search_decoder, model_type):
    greedy_metrics = {
        'cer': 0,
        'wer': 0
    }

    bs_metrics = {
        'cer': 0,
        'wer': 0
    }

    num_texts = 0

    results = []
    for batch in tqdm(test_loader, desc='Evaluating model'):
        if model_type == "ctc":
            logprobs = model(batch['specs'].to(device), 'test')
        else:
            batch['specs'] = batch['specs'].to(device)
            batch['specs_len'] = torch.LongTensor(batch['specs_len']).to(device)
            logprobs = model(batch, 'test')

        argmax_texts = greedy_decoder.decode(model, logprobs, batch['specs_len'])
        bs_texts = beam_search_decoder.decode(model, batch, batch['specs_len'], return_best=False)

        for argmax_text, bs_text, target in zip(argmax_texts, bs_texts, batch['text']):
            num_texts += 1

            record = {
                'ground_truth': target,
                'pred_text_argmax': argmax_text,
                'pred_beam_search': bs_text[:10]
            }
            results.append(record)
            greedy_metrics['cer'] += calc_cer(argmax_text, target)
            greedy_metrics['wer'] += calc_wer(argmax_text, target)
            bs_metrics['cer'] += calc_cer(bs_text[0], target)
            bs_metrics['wer'] += calc_wer(bs_text[0], target)

    with open(out_file, 'w+') as f:
        dump(results, f, indent=2)

    print(f"Greedy Decoding CER: {greedy_metrics['cer'] / num_texts}")
    print(f"Greedy Decoding WER: {greedy_metrics['wer'] / num_texts}")
    print(f"Beam Search Decoding CER: {bs_metrics['cer'] / num_texts}")
    print(f"Beam Search Decoding WER: {bs_metrics['wer'] / num_texts}")


if __name__ == '__main__':
    opts = parse_args()

    with open(opts.c, 'r') as f:
        config = load(f)

    Path(config['weights_dir']).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    download_checkpoint(config['checkpoint_params'])
    state = torch.load(config['checkpoint_params']['checkpoint_path'])

    # create test loader and tokenizer
    test_dir = opts.t
    if test_dir[-1] == '/':
        test_dir = test_dir[:-1]

    test_loader, tokenizer = make_test_loader(config, state['tokenizer'], test_dir)

    # load model
    if config['model']['constructor'] == 'LAS':
        config['model']['args']['bos_idx'] = tokenizer.bos_token_id
        config['model']['args']['padding_idx'] = tokenizer.eps_token_id

    model = make_generic('model', config['model']).to(config['device'])
    model.load_state_dict(state['model'])
    model.eval()

    # create text decoders

    config['text_decoders']['greedy']['args']['tokenizer'] = tokenizer

    if not exists(config['text_decoders']['beam_search']['args']['lm_weight_path']):
        gdown.download(config['text_decoders']['beam_search']['lm_link'],
                       config['text_decoders']['beam_search']['args']['lm_weight_path'])

    config['text_decoders']['beam_search']['args']['tokenizer'] = tokenizer

    greedy_decoder = make_generic('text_decoder', config['text_decoders']['greedy'])
    beam_search_decoder = make_generic('text_decoder', config['text_decoders']['beam_search'])

    eval_model(opts.o, model, config['device'], test_loader, greedy_decoder, beam_search_decoder, 'seq2seq')
