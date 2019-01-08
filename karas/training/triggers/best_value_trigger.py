import operator

import karas.reporter as reporter
from karas.training.triggers import utils


class BestValueTrigger(object):
    def __init__(self, key, compare, trigger=(1, 'epoch')):
        self.key = key
        self.best_value = None
        self.interval_trigger = utils.get_trigger(trigger)
        self.compare = compare

        self._init_summary()

    def __call__(self, trainer):
        observation = trainer.observation
        summary = self._summary

        for key in observation.keys():
            if self.key in key:
                summary.add({self.key: observation[key]})

        if not self.interval_trigger(trainer):
            return False

        stats = summary.compute_mean()
        value = float(stats[self.key])  # copy to CPU
        self._init_summary()

        if self.best_value is None or self.compare(self.best_value, value):
            self.best_value = value
            return True

        return False

    def _init_summary(self):
        self._summary = reporter.DictSummary()


class MaxValueTrigger(BestValueTrigger):
    def __init__(self, key, trigger=(1, 'epoch')):
        super(MaxValueTrigger, self).__init__(key, operator.gt, trigger)


class MinValueTrigger(BestValueTrigger):
    def __init__(self, key, trigger=(1, 'epoch')):
        super(MinValueTrigger, self).__init__(key, operator.lt, trigger)
