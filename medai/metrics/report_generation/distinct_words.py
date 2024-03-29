from collections import Counter
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

from medai.utils.nlp import PAD_IDX

class DistinctWords(Metric):
    """Counts amount of different words generated."""
    def __init__(self, ignore_pad=True, output_transform=lambda x: x, device=None):
        self.ignore_pad = ignore_pad

        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self.words_seen = Counter()

        super().reset()

    @reinit__is_reduced
    def update(self, output):
        """Update on each step.

        output:
            clean_reports -- list of lists of generated words,
                shape (batch_size, n_words_per_report)
            ground_truth -- unused
        """
        clean_reports, _ = output

        for report in clean_reports:
            for word in report:
                self.words_seen[word] += 1

    @sync_all_reduce('words_seen')
    def compute(self):
        if self.ignore_pad and PAD_IDX in self.words_seen:
            remove_ignored = 1
        else:
            remove_ignored = 0

        return len(self.words_seen) - remove_ignored
