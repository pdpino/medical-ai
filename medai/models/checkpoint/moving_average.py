from collections import deque
import logging

import numpy as np

LOGGER = logging.getLogger(__name__)

class SimpleMovingAverage:
    def __init__(self, n=10):
        assert n is not None

        self.items = deque(maxlen=n)

    def next(self, item):
        # appends on the right, drops on the left
        self.items.append(item)

        return np.mean(self.items)

class ExpMovingAverage:
    """Exponential moving average.

    Could have used RunningAverage from ignite, but realized too late."""
    def __init__(self, weight=0.7):
        assert weight > 0
        assert weight < 1

        self.weight = weight
        self.current = None

    def next(self, item):
        if self.current is None:
            self.current = item
        else:
            self.current = self.weight * self.current + (1-self.weight) * item

        return self.current

_MOVING_AVERAGES = {
    'simple': SimpleMovingAverage,
    'exp': lambda kw: ExpMovingAverage(weight=1-kw['weight']), # OLD: deprecated
    'exp-fixed': ExpMovingAverage,
}

AVAILABLE_MOVING_AVERAGES = list(_MOVING_AVERAGES)

def create_moving_average(mode, **kwargs):
    if mode not in _MOVING_AVERAGES:
        raise Exception(f'No such moving-average mode: {mode}')

    if mode == 'exp':
        LOGGER.error('Moving average mode %s is deprecated', mode)

    return _MOVING_AVERAGES[mode](**kwargs)
