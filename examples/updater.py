import torch.nn as nn

import karas.training.updater as updater


class Updater(updater.Updater):
    def __init__(self, model, **kwargs):
        super(Updater, self).__init__(**kwargs)
        self.model = model
        self.criterion = nn.NLLLoss().to(self.device)

    def converter(self, batch):
        input, target = batch
        input, target = input.to(self.device), target.to(self.device)
        return input, target

    def update(self, batch):
        input, target = self.converter(batch)

        output = self.model(input)
        loss = self.criterion(output, target)

        self.optimizers['net'].zero_grad()
        loss.backward()
        self.optimizers['net'].step()

        with self.reporter.scope('scalar'):
            self.reporter.report({'loss': loss.item()})

        with self.reporter.scope('images'):
            self.reporter.report({'input': input})
