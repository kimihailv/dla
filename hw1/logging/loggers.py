import wandb


class WandbLogger:
    def __init__(self, **run_kwargs):
        self.run = wandb.init(**run_kwargs)
        columns = ['wav', 'src', 'tgt']
        self.tables = {
            'train': wandb.Table(columns=columns),
            'val': wandb.Table(columns=columns),
            'test': wandb.Table(columns=columns)
        }

    def log(self, data):
        self.run.log(data)

    def add_row(self, wav, src, tgt, split):
        wav = wandb.Audio(wav, sample_rate=wandb.config['dataset_params']['preprocess']['sr'])
        self.tables[split].add_data(wav, src, tgt)

    def push_table(self, split):
        self.run.log({f'{split}_examples': self.tables[split]})

    def set_summary(self, data):
        for k, v in data.items():
            self.run.summary[k] = v

    def watch(self, **kwargs):
        self.run.watch(**kwargs)

    def finish(self):
        self.run.finish()
