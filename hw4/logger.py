import wandb


class WandbLogger:
    def __init__(self, **run_kwargs):
        self.run = wandb.init(**run_kwargs)
        self.prev_tables = {'train': None, 'val': None, 'test': None}
        self.cur_tables = {'train': None, 'val': None, 'test': None}

    def log(self, data):
        self.run.log(data)

    def add_row(self, epoch, gen_wav, tgt_wav, gen_mel_spec, tgt_mel_spec, split):
        columns = ['epoch', 'gen_wav', 'tgt_wav', 'gen_mel_spec', 'tgt_mel_spec']
        history = []
        gen_wav = wandb.Audio(gen_wav, sample_rate=22050)
        tgt_wav = wandb.Audio(tgt_wav, sample_rate=22050)
        gen_spec = wandb.Image(gen_mel_spec)
        tgt_spec = wandb.Image(tgt_mel_spec)

        if self.cur_tables[split] is None:
            if self.prev_tables[split] is not None:
                for _, row in self.prev_tables[split].iterrows():
                    history.append(row)
            history.append([epoch, gen_wav, tgt_wav,
                            gen_spec, tgt_spec])
            self.cur_tables[split] = wandb.Table(data=history, columns=columns)
        else:
            self.cur_tables[split].add_data(epoch, gen_wav, tgt_wav,
                                            gen_spec, tgt_spec)

    def push_table(self, split):
        self.run.log({f'{split}_examples': self.cur_tables[split]})
        self.prev_tables[split] = self.cur_tables[split]
        self.cur_tables[split] = None

    def set_summary(self, data):
        for k, v in data.items():
            self.run.summary[k] = v

    def watch(self, **kwargs):
        self.run.watch(**kwargs)

    def finish(self):
        self.run.finish()