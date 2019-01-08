from karas.training import extension


class LrScheduler(extension.Extension):
    def __init__(self, scheduler):
        self._scheduler = scheduler

    def __call__(self, trainer):
        self._scheduler.step()
