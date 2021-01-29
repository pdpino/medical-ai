import random
import logging

LOGGER = logging.getLogger(__name__)

class CircularShuffledList:
    """Implements a circular-list that shuffles their items in each lap."""
    def __init__(self, items):
        self.items = list(items)

        if len(self.items) == 0:
            LOGGER.warning('Empty shuffled-list is ill defined')

        self.reset()

    def __len__(self):
        return len(self.items)

    def reset(self):
        self._index = 0
        random.shuffle(self.items)

    def get_next(self):
        if len(self.items) == 0:
            return None

        if self._index >= len(self.items):
            self.reset()

        item = self.items[self._index]

        self._index += 1
        return item

    def __repr__(self):
        return self.items.__repr__()
