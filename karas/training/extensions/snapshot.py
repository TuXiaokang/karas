import os
import shutil

import torch

from karas.training import extension
from karas.training import utils


class Snapshot(extension.Extension):
    """
    if target == None, save the trainer
    """

    def __init__(self, target=None, filename='snapshot_iter_{.iteration}.pth'):
        self._tmpl = filename
        self._tget = target

    def __call__(self, trainer):
        device = trainer.updater.device

        fn = self._tmpl.format(trainer)
        prefix = 'tmp' + fn

        with utils.tempdir(prefix=prefix, dir=trainer.out) as tmpdir:
            tmppath = os.path.join(tmpdir, fn)

            if self._tget is None:
                trainer.serialize(tmppath)
            else:
                torch.save(self._tget.cpu(), tmppath)

            shutil.move(tmppath, os.path.join(trainer.out, fn))

        if self._tget is not None:
            self._tget.to(device)
