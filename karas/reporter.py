import collections
import contextlib
import math
import typing as tp  # NOQA

import numpy
import six
import torch

SCALAR = 100
IMAGE = 200
VIDEO = 300
TEXT = 400


class Reporter(object):
    """
    reporter
    See the following example::
        >>> reporter = Reporter()
        >>> with reporter.scope('loss'):
        ...     reporter.report({'x':1})
        ...
        >>> reporter.observation
        {'loss/x': 1}
    """

    def __init__(self):
        self.namespace = ''
        self.sep = '/'
        self.observation = {}

    @contextlib.contextmanager
    def scope(self, namespace):
        old = self.namespace
        if self.namespace != '':
            self.namespace = old + self.sep + namespace
        else:
            self.namespace = namespace
        self.__enter__()
        yield self
        self.__exit__(None, None, None)
        self.namespace = old

    def report(self, values):
        for key, value in six.iteritems(values):
            if self.namespace != '':
                name = '%s/%s' % (self.namespace, key)
            else:
                name = key
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
            self.observation[name] = value

    def reset(self):
        self.namespace = ''
        self.observation = {}

    def __enter__(self):
        _reporters.append(self)
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        _reporters.pop()
        pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state['namespace'] = ''
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


_reporters = []  # type: tp.Optional[tp.List[Reporter]]


def get_current_reporter():
    """Returns the current reporter object."""
    return _reporters[-1]


def report(values, observer=None):
    """Reports observed values with the current reporter object.

    Any reporter object can be set current by the ``with`` statement. This
    function calls the :meth:`Reporter.report` method of the current reporter.
    If no reporter object is current, this function does nothing.

    .. admonition:: Example

       The most typical example is a use within links and chains. Suppose that
       a link is registered to the current reporter as an observer (for
       example, the target link of the optimizer is automatically registered to
       the reporter of the :class:`~chainer.training.Trainer`). We can report
       some values from the link as follows::

          class MyRegressor(chainer.Chain):
              def __init__(self, predictor):
                  super(MyRegressor, self).__init__(predictor=predictor)

              def __call__(self, x, y):
                  # This chain just computes the mean absolute and squared
                  # errors between the prediction and y.
                  pred = self.predictor(x)
                  abs_error = F.sum(abs(pred - y)) / len(x)
                  loss = F.mean_squared_error(pred, y)

                  # Report the mean absolute and squared errors.
                  chainer.report({
                      'abs_error': abs_error,
                      'squared_error': loss,
                  }, self)

                  return loss

       If the link is named ``'main'`` in the hierarchy (which is the default
       name of the target link in the
       :class:`~chainer.training.updaters.StandardUpdater`),
       these reported values are
       named ``'main/abs_error'`` and ``'main/squared_error'``. If these values
       are reported inside the :class:`~chainer.training.extensions.Evaluator`
       extension, ``'validation/'`` is added at the head of the link name, thus
       the item names are changed to ``'validation/main/abs_error'`` and
       ``'validation/main/squared_error'`` (``'validation'`` is the default
       name of the Evaluator extension).

    Args:
        values (dict): Dictionary of observed values.
        observer: Observer object. Its object ID is used to retrieve the
            observer name, which is used as the prefix of the registration name
            of the observed value.

    """
    if _reporters:
        current = _reporters[-1]
        current.report(values, observer)


@contextlib.contextmanager
def report_scope(observation):
    """Returns a report scope with the current reporter.

    This is equivalent to ``get_current_reporter().scope(observation)``,
    except that it does not make the reporter current redundantly.

    """
    current = _reporters[-1]
    old = current.observation
    current.observation = observation
    yield
    current.observation = old


# def _get_device(x):
#     if numpy.isscalar(x):
#         return cuda.DummyDevice
#     else:
#         return cuda.get_device_from_array(x)
#

class Summary(object):
    """Online summarization of a sequence of scalars.

    Summary computes the statistics of given scalars online.

    """

    def __init__(self):
        self._x = 0.0
        self._x2 = 0.0
        self._n = 0

    def add(self, value, weight=1):
        """Adds a scalar value.

        Args:
            value: Scalar value to accumulate. It is either a NumPy scalar or
                a zero-dimensional array (on CPU or GPU).
            weight: An optional weight for the value. It is a NumPy scalar or
                a zero-dimensional array (on CPU or GPU).
                Default is 1 (integer).

        """
        # if isinstance(value, chainerx.ndarray):
        #     # ChainerX arrays does not support inplace assignment if it's
        #     # connected to the backprop graph.
        #     value = value.as_grad_stopped()

        # with _get_device(value):
        self._x += weight * value
        self._x2 += weight * value * value
        self._n += weight

    def compute_mean(self):
        """Computes the mean."""
        x, n = self._x, self._n
        # with _get_device(x):
        return x / n

    def make_statistics(self):
        """Computes and returns the mean and standard deviation values.

        Returns:
            tuple: Mean and standard deviation values.

        """
        x, n = self._x, self._n
        # with _get_device(x):
        mean = x / n
        var = self._x2 / n - mean * mean
        std = math.sqrt(var)
        return mean, std

    # def serialize(self, serializer):
    #     try:
    #         self._x = serializer('_x', self._x)
    #         self._x2 = serializer('_x2', self._x2)
    #         self._n = serializer('_n', self._n)
    #     except KeyError:
    #         warnings.warn('The previous statistics are not saved.')


class DictSummary(object):
    """Online summarization of a sequence of dictionaries.

    ``DictSummary`` computes the statistics of a given set of scalars online.
    It only computes the statistics for scalar values and variables of scalar
    values in the dictionaries.

    """

    def __init__(self):
        self._summaries = collections.defaultdict(Summary)

    def add(self, d):
        """Adds a dictionary of scalars.

        Args:
            d (dict): Dictionary of scalars to accumulate. Only elements of
               scalars, zero-dimensional arrays, and variables of
               zero-dimensional arrays are accumulated. When the value
               is a tuple, the second element is interpreted as a weight.

        """

        summaries = self._summaries
        for k, v in six.iteritems(d):
            w = 1
            if isinstance(v, tuple):
                w = v[1]
                v = v[0]
                if isinstance(w, torch.Tensor):
                    w = w.cpu().numpy()
                if not numpy.isscalar(w) and not getattr(w, 'ndim', -1) == 0:
                    raise ValueError('Given weight to {} was not scalar.'.format(k))
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            if numpy.isscalar(v) or getattr(v, 'ndim', -1) == 0:
                summaries[k].add(v, weight=w)

    def compute_mean(self):
        """Creates a dictionary of mean values.

        It returns a single dictionary that holds a mean value for each entry
        added to the summary.

        Returns:
            dict: Dictionary of mean values.

        """
        return {name: summary.compute_mean()
                for name, summary in six.iteritems(self._summaries)}

    def make_statistics(self):
        """Creates a dictionary of statistics.

        It returns a single dictionary that holds mean and standard deviation
        values for every entry added to the summary. For an entry of name
        ``'key'``, these values are added to the dictionary by names ``'key'``
        and ``'key.std'``, respectively.

        Returns:
            dict: Dictionary of statistics of all entries.

        """
        stats = {}
        for name, summary in six.iteritems(self._summaries):
            mean, std = summary.make_statistics()
            stats[name] = mean
            stats[name + '.std'] = std

        return stats

    # def serialize(self, serializer):
    #     if isinstance(serializer, serializer_module.Serializer):
    #         names = list(self._summaries.keys())
    #         serializer('_names', json.dumps(names))
    #         for index, name in enumerate(names):
    #             self._summaries[name].serialize(
    #                 serializer['_summaries'][str(index)])
    #     else:
    #         self._summaries.clear()
    #         try:
    #             names = json.loads(serializer('_names', ''))
    #         except KeyError:
    #             warnings.warn('The names of statistics are not saved.')
    #             return
    #         for index, name in enumerate(names):
    #             self._summaries[name].serialize(
    #                 serializer['_summaries'][str(index)])
