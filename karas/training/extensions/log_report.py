import json
import os
import shutil

import six

from karas import compare_key
from karas import reporter
from karas.training import extension, utils
from karas.training.triggers.utils import get_trigger


class LogReport(extension.Extension):
    """Trainer extension to output the accumulated results to a log file.

    This extension accumulates the observations of the trainer to
    :class:`~chainer.DictSummary` at a regular interval specified by a supplied
    trigger, and writes them into a log file in JSON format.

    There are two triggers to handle this extension. One is the trigger to
    invoke this extension, which is used to handle the timing of accumulating
    the results. It is set to ``1, 'iteration'`` by default. The other is the
    trigger to determine when to emit the result. When this trigger returns
    True, this extension appends the summary of accumulated values to the list
    of past summaries, and writes the list to the log file. Then, this
    extension makes a new fresh summary object which is used until the next
    time that the trigger fires.

    It also adds some entries to each result dictionary.

    - ``'epoch'`` and ``'iteration'`` are the epoch and iteration counts at the
      output, respectively.
    - ``'elapsed_time'`` is the elapsed time in seconds since the training
      begins. The value is taken from :attr:`Trainer.elapsed_time`.

    Args:
        keys (iterable of strs): Keys of values to accumulate. If this is None,
            all the values are accumulated and output to the log file.
        trigger: Trigger that decides when to aggregate the result and output
            the values. This is distinct from the trigger of this extension
            itself. If it is a tuple in the form ``<int>, 'epoch'`` or
            ``<int>, 'iteration'``, it is passed to :class:`IntervalTrigger`.
        postprocess: Callback to postprocess the result dictionaries. Each
            result dictionary is passed to this callback on the output. This
            callback can modify the result dictionaries, which are used to
            output to the log file.
        log_name (str): Name of the log file under the output directory. It can
            be a format string: the last result dictionary is passed for the
            formatting. For example, users can use '{iteration}' to separate
            the log files for different iterations. If the log name is None, it
            does not output the log to any file.

    """

    def __init__(self, keys=None, trigger=(1, 'epoch'), postprocess=None, log_name='log'):
        self._keys = keys
        self._trigger = get_trigger(trigger)
        self._postprocess = postprocess
        self._log_name = log_name
        self._log = []

        self._init_summary()

    def __call__(self, trainer):
        # accumulate the observations
        keys = self._keys
        observation = trainer.observation
        summary = self._summary

        if keys is None:
            summary.add(observation)
        else:
            for tag in observation.keys():
                for key in keys:
                    if compare_key(key, tag):
                        summary.add({key: observation[tag]})

        if not self._trigger(trainer):
            # output the result
            stats = self._summary.compute_mean()
            stats_cpu = {}
            for name, value in six.iteritems(stats):
                stats_cpu[name] = float(value)  # copy to CPU

            stats_cpu['epoch'] = trainer.epoch
            stats_cpu['iteration'] = trainer.iteration
            stats_cpu['elapsed_time'] = trainer.elapsed_time

            if self._postprocess is not None:
                self._postprocess(stats_cpu)

            self._log.append(stats_cpu)

            # write to the log file
            if self._log_name is not None:
                log_name = self._log_name.format(**stats_cpu)
                with utils.tempdir(prefix=log_name, dir=trainer.out) as tempd:
                    path = os.path.join(tempd, 'log.json')
                    with open(path, 'w') as f:
                        json.dump(self._log, f, indent=4)

                    new_path = os.path.join(trainer.out, log_name)
                    shutil.move(path, new_path)

            # reset the summary for the next output
            self._init_summary()

    @property
    def log(self):
        """The current list of observation dictionaries."""
        return self._log

    def _init_summary(self):
        self._summary = reporter.DictSummary()
