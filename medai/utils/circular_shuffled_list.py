import random
import logging

LOGGER = logging.getLogger(__name__)

class CircularShuffledList:
    """Implements a circular-list that shuffles their items in each lap.

    Has two usages (do not mix them!). In both cases, each time you iterate over
    the list, a different order will be given

    1. Iter:
        for item in circular_list:
            # Do something with item
            pass

    2. get_next() or next()
        for _ in range(len(circular_list)):
            item = next(circular_list)
            # or: item = circular_list.get_next()
            # Do something with item

    If zip is used with multiple lists, zip_longest() should be used
    """
    def __init__(self, items):
        self.items = list(items)

        if len(self.items) == 0:
            LOGGER.warning('Empty shuffled-list is ill defined')

        self.reset()

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        for index in range(len(self.items)):
            item = self.items[index]
            yield item

        self.reset()

    def reset(self):
        self._index = 0
        random.shuffle(self.items)

    def __next__(self):
        return self.get_next()

    def get_next(self):
        # TODO: delete this method, keep only __next__
        if len(self.items) == 0:
            return None

        if self._index >= len(self.items):
            self.reset()

        item = self.items[self._index]

        self._index += 1
        return item

    def __repr__(self):
        return self.items.__repr__()
