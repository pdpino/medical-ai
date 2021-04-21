import logging
from ignite.metrics import Metric

LOGGER = logging.getLogger(__name__)

class HitCounterMetric(Metric):
    """Metric to capture Labeler hit-counters."""
    def __init__(self, labeler=None, device=None):
        self._labeler = labeler

        dummy_transform = lambda _: (None, None)
        super().__init__(output_transform=dummy_transform, device=device)

    def reset(self):
        self._labeler.hit_counter.reset()

        super().reset()

    def update(self, unused_output):
        pass

    def compute(self):
        n_misses, n_unique_misses, n_total = self._labeler.hit_counter.compute()

        percentage = n_misses / n_total

        return n_misses, n_unique_misses, percentage
