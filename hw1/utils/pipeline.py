import torch
from .factory import make_generic, make_aug, make_dataset, make_mel_transform, Compose
from ..data.datasets import Collator
from ..eval.decoding import TextDecoder
from ..eval.metrics import calc_cer, calc_wer
from json import load
from tqdm import tqdm
from os import environ
from copy import deepcopy
import numpy as np


class Pipeline:
    def __init__(self,
                 model_params,
                 dataset_params,
                 optimizer_params,
                 scheduler_params,
                 logger_params,
                 training_params):
        self.tokenizer = None
        self.training_params = training_params
        self.device = training_params['device']
        self.train_loader, tokenizer = self.make_loader(dataset_params['train'], dataset_params['preprocess'])
        self.tokenizer = tokenizer
        self.val_loader, _ = self.make_loader(dataset_params['val'], dataset_params['preprocess'])
        self.test_loader, _ = self.make_loader(dataset_params['test'], dataset_params['preprocess'])

        model_params['args']['voc_size'] = len(self.tokenizer)
        self.model = make_generic('model', model_params).to(self.device)
        optimizer_params['args']['params'] = self.model.parameters()
        self.optimizer = make_generic('optimizer', optimizer_params)
        scheduler_params['args']['optimizer'] = self.optimizer
        self.scheduler = make_generic('scheduler', scheduler_params)
        logger_params['models'] = self.model
        self.logger = make_generic('logger', logger_params)

        if self.training_params['criterion']['constructor'] == 'CTCLoss':
            self.training_params['criterion']['args']['blank'] = self.tokenizer.eps_token_id
        self.criterion = make_generic('loss', training_params['criterion'])
        self.text_decoder = TextDecoder(tokenizer)

    def make_loader(self, dataset_params, preprocess_params):
        wav_transform = None
        if len(dataset_params['aug']) > 0:
            wav_transform = Compose(*[make_aug(aug_params) for aug_params in dataset_params['aug']])
        mel_transform = make_mel_transform(dataset_params["mel_transform"])
        dataset, tokenizer = make_dataset(dataset_params, preprocess_params, self.tokenizer)

        collator = Collator(wav_transform, mel_transform,
                            input_len_div_factor=preprocess_params['input_len_div_factor'])
        return torch.utils.data.DataLoader(dataset, collate_fn=collator, **dataset_params['loader']), tokenizer

    def train_one_epoch(self, epoch_num):
        self.model.train()
        bar = tqdm(self.train_loader, position=0, leave=True, desc=f'Epoch #{epoch_num}')
        running_loss = 0
        num_samples = 0

        for idx, batch in enumerate(bar):
            self.optimizer.zero_grad()
            loss = self.model.calc_loss(batch, self.device, self.criterion)
            loss.backward()
            self.optimizer.step()

            loss_v = loss.item()
            self.logger.log({'train_iter_loss': loss_v, 'epoch': epoch_num, 'batch': idx})
            num_samples += 1
            running_loss += loss_v

        return running_loss / len(self.train_loader)

    @torch.no_grad()
    def eval(self, loader, mode):
        self.model.eval()
        samples_to_log = 5
        running_loss = 0
        cer = 0
        wer = 0
        num_samples = 0
        num_texts = 0

        for batch in loader:
            num_samples += 1
            loss, logprobs = self.model.calc_loss(batch, self.device, self.criterion, return_output=True)
            running_loss += loss.item()

            texts = self.text_decoder.decode(logprobs, batch['specs_len'])
            for src, tgt in zip(texts, batch['text']):
                num_texts += 1
                cer += calc_cer(src, tgt)
                wer += calc_wer(src, tgt)

            if samples_to_log > 0:
                sample_idx = np.random.choice(len(texts))
                src = texts[sample_idx]
                tgt = batch['text'][sample_idx]
                spec_len = batch['specs_len'][sample_idx]
                spec = batch['specs'][sample_idx][:, :spec_len].cpu().numpy()
                self.logger.add_row(batch['wavs'][sample_idx], spec, src, tgt, mode)
                samples_to_log -= 1

        self.logger.push_table(mode)
        return running_loss / num_samples, cer / num_texts, wer / num_texts

    def train(self):
        self.logger.watch(models=self.model)
        best_cer = 1
        best_wer = 1

        for epoch_num in range(self.training_params['total_epochs']):
            train_loss = self.train_one_epoch(epoch_num)
            self.scheduler.step()
            self.logger.log({'train_epoch_loss': train_loss,
                             'lr': self.scheduler.get_last_lr()[0],
                             'epoch': epoch_num})

            if epoch_num % self.training_params['eval_every'] == 0:
                train_loss, train_cer, train_wer = self.eval(self.train_loader, 'train')
                val_loss, val_cer, val_wer = self.eval(self.val_loader, 'val')

                self.logger.log({'train_cer': train_cer,
                                 'train_wer': train_wer,
                                 'val_loss': val_loss,
                                 'val_cer': val_cer,
                                 'val_wer': val_wer,
                                 'epoch': epoch_num
                                })

                if val_cer < best_cer:
                    best_cer = val_cer

                if val_wer < best_wer:
                    best_wer = val_wer

        test_loss, cer, wer = self.eval(self.test_loader, 'test')
        self.logger.set_summary({
            'test_loss': test_loss,
            'test_cer': cer,
            'test_wer': wer,
            'best_cer': best_cer,
            'best_wer': best_wer
        })

        self.logger.finish()

    @classmethod
    def from_config_file(cls, path_to_config):
        with open(path_to_config, 'r') as config:
            pipeline_params = load(config)

        pipeline_params['logger_params']['args']['config'] = deepcopy(pipeline_params)
        environ['WANDB_API_KEY'] = pipeline_params['logger_params']['api_token']
        return cls(**pipeline_params)
