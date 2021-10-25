import torch
import gdown
from os.path import exists
from argparse import ArgumentParser
from factory import make_generic, make_mel_transform
from ..data.datasets import TestDataset, Collator
from ..eval.metrics import *
from json import load, dump
from torch.utils.data import DataLoader
from tqdm import tqdm


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
    if exists(config['checkpoint_dir']):
        return

    gdown.download(config['checkpoint_link'], config['checkpoint_dir'])
    if 'bpe_link' in config:
        gdown.download(config['bpe_link'], config['bpe_dir'])


def make_test_loader(config, tokenizer_state, test_data_dir):
    tokenizer = make_generic('tokenizer', config['tokenizer_params'])
    tokenizer.load(tokenizer_state)
    dataset = TestDataset(tokenizer, test_data_dir, config['dataset_params']['sr'])
    mel_transform = make_mel_transform(config['dataset_params']['mel_transform'])
    collator = Collator(wav_transform=None, mel_transform=mel_transform)

    return DataLoader(dataset, batch_size=32, collate_fn=collator), tokenizer


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
        bs_texts = beam_search_decoder.decode(model, batch, batch['specs_len'])

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

    # load checkpoint
    download_checkpoint(config['checkpoint_params'])
    state = torch.load(config['checkpoint_params']['checkpoint_dir'])

    # create test loader and tokenizer
    test_dir = opts.t
    if test_dir[-1] == '/':
        test_dir = test_dir[:-1]

    test_loader, tokenizer = make_test_loader(config, state['tokenizer'], test_dir)

    # load model
    if config['model']['constructor'] == 'LAS':
        config['model']['args']['bos_idx'] = tokenizer.bos_token_id
        config['model']['args']['padding_idx'] = tokenizer.eps_token_id

    model = make_generic('model', config['model'])
    model.load_state_dict(state['model'])
    model.eval()

    # create text decoders

    config['text_decoders']['greedy']['args']['tokenizer'] = tokenizer
    config['text_decoders']['beam_search']['args']['tokenizer'] = tokenizer

    greedy_decoder = make_generic('text_decoder', config['text_decoders']['greedy'])
    beam_search_decoder = make_generic('text_decoder', config['text_decoders']['beam_search'])

    eval_model(opts.o, model, config['device'], test_loader, greedy_decoder, beam_search_decoder, 'seq2seq')