import os
import time

import karas
from karas.iterators.iterator import Iterator
from karas.reporter import Reporter
from karas.training.triggers.utils import get_trigger

# Select the best-resolution timer function
try:
    _get_time = time.perf_counter
except AttributeError:
    if os.name == 'nt':
        _get_time = time.clock
    else:
        _get_time = time.time


class Trainer(object):
    def __init__(self, updater, stop_trigger, loaders, out='output'):
        self._reporter = Reporter()

        self._updater = updater
        self._updater(self)

        self._stop_trigger = get_trigger(stop_trigger) if isinstance(stop_trigger, tuple) else stop_trigger

        assert 'train' in loaders.keys(), 'loader has no "train" attribute.'

        self._loaders = loaders
        self._iterators = {}
        for key, value in self._loaders.items():
            repeat = True if 'train' == key else False
            self._iterators[key] = Iterator(dataloader=value, repeat=repeat)

        self._done = False
        self._start_at = None
        self._snapshot_elapsed_time = 0.0
        self._final_elapsed_time = None

        self._extensions = {}
        self._out = out

    def get_loader(self, name):
        return self._loaders[name]

    def get_iterator(self, name):
        return self._iterators[name]

    @property
    def iteration(self):
        return self._iterators['train'].iteration

    @property
    def epoch(self):
        return self._iterators['train'].epoch

    @property
    def epoch_detail(self):
        return self._iterators['train'].epoch_detail

    @property
    def previous_epoch_detail(self):
        return self._iterators['train'].previous_epoch_detail

    @property
    def position(self):
        return self._iterators['train'].position

    @property
    def updater(self):
        return self._updater

    @property
    def stop_trigger(self):
        return self._stop_trigger

    @property
    def reporter(self):
        return self._reporter

    @property
    def observation(self):
        return self._reporter.observation

    @property
    def elapsed_time(self):
        """Total time used for the training.
        The time is in seconds. If the training is resumed from snapshot, it
        includes the time of all the previous training to get the current
        state of the trainer.
        """
        if self._done:
            return self._final_elapsed_time
        if self._start_at is None:
            raise RuntimeError('training has not been started yet')
        return _get_time() - self._start_at + self._snapshot_elapsed_time

    @property
    def out(self):
        return self._out

    def get_extension(self, name):
        """Returns the extension of a given name.
        Args:
            name (str): Name of the extension.
        Returns:
            Extension.
        """
        extensions = self._extensions
        if name in extensions:
            return extensions[name]
        else:
            raise ValueError('extension %s not found' % name)

    def extend(self, extension, name=None, trigger=None, priority=None, **kwargs):
        if name is None:
            name = extension.default_name
        if trigger is not None:
            extension.trigger = trigger
        if priority is not None:
            extension.priority = priority

        modified_name = name
        ordinal = 0
        while modified_name in self._extensions:
            ordinal += 1
            modified_name = '%s_%d' % (name, ordinal)

        extension.name = modified_name
        self._extensions[modified_name] = extension

    def run(self):

        try:
            os.makedirs(self.out)
        except OSError:
            pass

        # sort extensions by priorities
        extension_order = sorted(self._extensions.keys(), key=lambda name: self._extensions[name].priority,
                                 reverse=True)
        extensions = [(name, self._extensions[name]) for name in extension_order]

        for _, entry in extensions:
            initialize = getattr(entry, 'initialize', None)
            if initialize:
                with self.reporter.scope('test'):
                    initialize(self)

        self._start_at = _get_time()

        try:
            while not self._stop_trigger(self) and self._iterators['train'].has_next():

                # self.reporter.reset()
                batch = next(self._iterators['train'])

                with self.reporter.scope('train'):
                    self._updater.update(batch)

                with self.reporter.scope('test'):
                    for name, entry in extensions:
                        if entry.trigger(self):
                            entry(self)

        except Exception as e:
            print('exception: {}'.format(e))

        finally:
            for _, entry in extensions:
                finalize = getattr(entry, 'finalize', None)
                if finalize:
                    finalize()

        self._final_elapsed_time = self.elapsed_time
        self._done = True
        print('Training Finished')

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def serialize(self, filename):
        karas.serialize(self, filename)

    def deserialize(self, filename):
        self = karas.deserialize(os.path.join(self.out, filename))
