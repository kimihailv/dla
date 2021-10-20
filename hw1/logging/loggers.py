import wandb


class WandbLogger:
    def __init__(self, **run_kwargs):
        self.run = wandb.init(**run_kwargs)
        self.prev_tables = {'train': None, 'val': None, 'test': None}
        self.cur_tables = {'train': None, 'val': None, 'test': None}

    def log(self, data):
        self.run.log(data)

    def add_row(self, epoch, wav, mel_spec, src, tgt, split):
        columns = ['epoch', 'wav', 'mel_spec', 'src', 'tgt']
        history = []
        sr = self.run.config['dataset_params']['common']['sr']
        wav = wandb.Audio(wav, sample_rate=sr)
        spec = wandb.Image(mel_spec)

        if self.cur_tables[split] is None:
            if self.prev_tables[split] is not None:
                for _, row in self.prev_tables[split].iterrows():
                    history.append(row)
            history.append([epoch, wav, spec, src, tgt])
            self.cur_tables[split] = wandb.Table(data=history, columns=columns)
        else:
            self.cur_tables[split].add_data(epoch, wav, spec, src, tgt)

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