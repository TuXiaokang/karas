import torch
from extension import Eval
from net import Net
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from updater import Updater

import karas
# from karas.dataloader import DataLoader
from karas.training.extension import *
from karas.training.extensions import LogReport
from karas.training.extensions import LrObserver
from karas.training.extensions import LrScheduler
from karas.training.extensions import PrintReport
from karas.training.extensions import ProgressBar
from karas.training.extensions import Snapshot
from karas.training.extensions import TensorBoard
from karas.training.trainer import Trainer
from karas.training.triggers import EarlyStoppingTrigger
from karas.training.triggers import MaxValueTrigger

resume = False

loaders = {}
loaders['train'] = DataLoader(MNIST('../data', train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        # transforms.Normalize((0.1307,), (0.3081,))
                                    ])), batch_size=16)

loaders['test'] = DataLoader(MNIST('../data', train=False, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       # transforms.Normalize((0.1307,), (0.3081,))
                                   ])))

keys = ['epoch', 'iteration', 'loss', 'test/net/lr', 'test/accuracy', 'elapsed_time']

device = torch.device('cpu')
net = Net().to(device)
opt = SGD(net.parameters(), lr=0.01, momentum=0.5)

stopper = EarlyStoppingTrigger(monitor='test/accuracy', patients=3, max_trigger=(10, 'epoch'))
updater = Updater(model=net, optimizers={'net': opt}, device=device)
trainer = Trainer(updater, stop_trigger=stopper, loaders=loaders)

# add extensions
trainer.extend(Snapshot(filename='snapshot_iter_{.iteration}.pkl'), priority=PRIORITY_READER, trigger=(1, 'epoch'))
trainer.extend(Snapshot(net, 'net_{.iteration}.pth'), priority=PRIORITY_READER, trigger=MaxValueTrigger(key='accuracy'))
trainer.extend(Eval(), priority=PRIORITY_WRITER, trigger=(1, 'epoch'))
trainer.extend(LogReport(keys), trigger=(100, 'iteration'))
trainer.extend(TensorBoard(), trigger=(10, 'iteration'))
trainer.extend(PrintReport(keys), trigger=(100, 'iteration'))
trainer.extend(LrObserver(), trigger=(100, 'iteration'))
trainer.extend(LrScheduler(lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)), trigger=(1, 'epoch'))
trainer.extend(ProgressBar(update_interval=10))

if resume:
    trainer = karas.deserialize('output/snapshot_iter_1000.pkl')

trainer.run()
