import wandb


class WandbLogger:
    def __init__(self, **run_kwargs):
        self.run = wandb.init(**run_kwargs)
        self.tables = {
            'train': wandb.Table(columns=['src', 'tgt']),
            'val': wandb.Table(columns=['src', 'tgt']),
            'test': wandb.Table(columns=['src', 'tgt'])
        }

    def log(self, data):
        self.run.log(data)

    def log_example(self, src, tgt, split):
        self.tables[split].add_data(src, tgt)

    def push_table(self, split):
        self.run.log({f'{split}_examples': self.tables[split]})

    def set_summary(self, data):
        for k, v in data.items():
            self.run.summary[k] = v

    def watch(self, **kwargs):
        self.run.watch(**kwargs)

    def finish(self):
        self.run.finish()
