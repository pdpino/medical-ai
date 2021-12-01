from collections import deque

import numpy as np

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
    def __init__(self, weight=0.6):
        assert weight > 0
        assert weight < 1

        self.weight = weight
        self.current = None

    def next(self, item):
        if self.current is None:
            self.current = item
        else:
            self.current = (1-self.weight) * self.current + self.weight * item

        return self.current

def create_moving_average(mode, **kwargs):
    if mode == 'simple':
        return SimpleMovingAverage(**kwargs)
    if mode == 'exp':
        return ExpMovingAverage(**kwargs)

    raise Exception(f'No such moving-average mode: {mode}')
