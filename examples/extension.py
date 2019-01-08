import torch

from karas.training import extension


class Eval(extension.Extension):
    def initialize(self, trainer):
        pass

    def __call__(self, trainer):
        updater = trainer.updater
        reporter = trainer.reporter
        loader = trainer.get_loader('test')
        correct = 0

        updater.model.eval()
        torch.set_grad_enabled(False)

        for batch in loader:
            input, target = updater.converter(batch)
            output = updater.model(input)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / len(loader.dataset)

        with reporter.scope('scalar'):
            reporter.report({'accuracy': accuracy})

        updater.model.train()
        torch.set_grad_enabled(True)

    def finalize(self):
        pass
