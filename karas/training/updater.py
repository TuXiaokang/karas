class Updater(object):
    def __init__(self, **kwargs):
        self.device = kwargs.pop('device')
        self.optimizers = kwargs.pop('optimizers')
        assert isinstance(self.optimizers, dict)

    def __call__(self, trainer):
        self.trainer = trainer
        self.reporter = trainer.reporter
        return self

    def get_optimizers(self):
        return self.optimizers

    def get_optimizer(self, name):
        if name in self.optimizers.keys():
            return self.optimizers[name]
        else:
            raise ValueError('No {} key in optimizers' % name)

    def converter(self, batch):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError
