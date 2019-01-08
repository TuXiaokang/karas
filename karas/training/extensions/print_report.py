import os
import sys

from karas.training import extension
from karas.training import utils
from karas.training.extensions import log_report as log_report_module


class PrintReport(extension.Extension):
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

    def __init__(self, entries, log_report='LogReport', out=sys.stdout):
        self._entries = entries
        self._log_report = log_report
        self._out = out

        self._log_len = 0  # number of observations already printed

        # format information
        entry_widths = [max(10, len(s)) for s in entries]

        header = '  '.join(('{:%d}' % w for w in entry_widths)).format(
            *entries) + '\n'
        self._header = header  # printed at the first call
        self._show_header = True

        templates = []
        for entry, w in zip(entries, entry_widths):
            templates.append((entry, '{:<%dg}  ' % w, ' ' * (w + 2)))
        self._templates = templates

    def __call__(self, trainer):

        out = self._out

        if self._show_header:
            out.write(self._header)
            self._show_header = False

        log_report = self._log_report
        if isinstance(log_report, str):
            log_report = trainer.get_extension(log_report)
        elif isinstance(log_report, log_report_module.LogReport):
            log_report(trainer)  # update the log report
        else:
            raise TypeError('log report has a wrong type %s' % type(log_report))
        log = log_report.log
        log_len = self._log_len
        while len(log) > log_len:
            # delete the printed contents from the current cursor
            # out.write('\033[1A')
            if os.name == 'nt':
                utils.erase_console(0, 0)
            else:
                out.write('\033[1A')
                out.write('\033[J')
            self._print(log[log_len])
            log_len += 1
        self._log_len = log_len

        if os.name != 'nt':
            out.write(self._header)


    def _print(self, observation):
        out = self._out
        for entry, template, empty in self._templates:
            if entry in observation:
                out.write(template.format(observation[entry]))
            else:
                out.write(empty)
        out.write('\n')
        if hasattr(out, 'flush'):
            out.flush()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_out']
        state['_log_len'] = 0
        state['_show_header'] = True
        return state

    def __setstate__(self, state):
        state['_out'] = sys.stdout
        self.__dict__.update(state)
