from ignite.metrics import Metric

from medai.utils import duration_to_str

class LabelerTimerMetric(Metric):
    """Metric to capture Labeler timers."""
    def __init__(self, labeler):
        self.labeler = labeler

    def reset(self):
        self.labeler.timer.reset()
        self.labeler.global_timer.reset()

    def update(self, unused_output):
        pass

    def compute(self):
        labeler_minutes = round(self.labeler.timer.total / 60, 1)
        global_minutes = round(self.labeler.global_timer.total / 60, 1)

        return f'{labeler_minutes}/{global_minutes}'