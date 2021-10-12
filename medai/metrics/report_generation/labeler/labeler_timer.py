import logging
from ignite.metrics import Metric

from medai.utils import duration_to_str

LOGGER = logging.getLogger(__name__)

class LabelerTimerMetric(Metric):
    """Metric to capture Labeler timers."""
    def __init__(self, labeler=None, device=None):
        self._labeler = labeler

        dummy_transform = lambda _: (None, None)
        super().__init__(output_transform=dummy_transform, device=device)

    def reset(self):
        self._labeler.timer.reset()
        self._labeler.global_timer.reset()

        super().reset()

    def update(self, unused_output):
        pass

    def compute(self):
        labeler_minutes = self._labeler.timer.total / 60
        global_minutes = self._labeler.global_timer.total / 60

        LOGGER.debug(
            'Labeler took: %s/%s',
            duration_to_str(self._labeler.timer.total),
            duration_to_str(self._labeler.global_timer.total),
        )

        if labeler_minutes == 0 and global_minutes == 0:
            # Empty value
            return -1

        return labeler_minutes
