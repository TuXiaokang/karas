from karas.training import extension


class LrObserver(extension.Extension):

    def initialize(self, trainer):
        self(trainer)

    def __call__(self, trainer):
        reporter = trainer.reporter
        optimizers = trainer.updater.get_optimizers()

        for key, optimizer in optimizers.items():
            lr = optimizer.param_groups[0]['lr']
            with reporter.scope('scalar'):
                reporter.report({'{}/lr'.format(key): lr})
