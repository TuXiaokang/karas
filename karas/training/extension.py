from karas.training.triggers.utils import get_trigger

PRIORITY_WRITER = 300
PRIORITY_EDITOR = 200
PRIORITY_READER = 100


class Extension(object):
    _trigger = get_trigger((1, 'iteration'))
    _priority = PRIORITY_READER
    _name = None

    def __call__(self, trainer):
        """
        connect to trainer
        :param trainer:
        :return:
        """
        raise NotImplementedError

    @property
    def default_name(self):
        return type(self).__name__

    @property
    def trigger(self):
        return self._trigger

    @trigger.setter
    def trigger(self, trigger):
        self._trigger = get_trigger(trigger)

    @property
    def priority(self):
        return self._priority

    @priority.setter
    def priority(self, priority):
        self._priority = priority

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name
