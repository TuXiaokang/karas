import tensorboardX as tbx

from karas import compare_key
from karas.training import extension


class TensorBoard(extension.Extension):
    """Trainer extension to print the accumulated results.

    This extension uses the log accumulated by a :class:`LogReport` extension
    to print specified entries of the log in a human-readable format.

    Args:
        entries (list of str): List of keys of observations to print.
        log_report (str or LogReport): Log report to accumulate the
            observations. This is either the name of a LogReport extensions
            registered to the trainer, or a LogReport instance to use
            internally.
        out: Stream to print the bar. Standard output is used by default.

    """

    def __init__(self, keys=None, out='logdir'):
        self._keys = keys
        self._out = out

    def initialize(self, trainer):
        self._writer = tbx.SummaryWriter(log_dir=self._out)

    def __call__(self, trainer):
        observation = trainer.observation
        epoch = trainer.epoch
        iteration = trainer.iteration

        for tag, value in observation.items():
            step = epoch if 'test' in tag else iteration

            haskey = False
            if self._keys is not None:
                for key in self._keys:
                    if compare_key(key, tag):
                        haskey = True
                        break;
                if not haskey:
                    continue

            if 'scalar' in tag:
                self._writer.add_scalar(tag, value, global_step=step)
            elif 'images' in tag:
                self._writer.add_image(tag, value.cpu().numpy(), global_step=step)
            pass

    def finalize(self):
        self._writer.close()

    def __getstate__(self):
        state = {}
        state['_keys'] = self._keys
        state['_out'] = self._out
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._writer = tbx.SummaryWriter(log_dir=self._out)
