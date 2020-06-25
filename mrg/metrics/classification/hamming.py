import torch
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class Hamming(Metric):
    """Computes Hamming loss (smaller is better).
    
    Using calculation described here: https://stats.stackexchange.com/a/168952/221232
    """
    @reinit__is_reduced
    def reset(self):
        self._result = 0
        self._n_samples = 0

        super().reset()

    @reinit__is_reduced
    def update(self, output):
        """Updates the metric state after a batch/epoch.
        
        output = (outputs, labels)
            - shape for both arrays: batch_size, n_labels
            - in binary format (rounded)
        """
        outputs, labels = output

        batch_size, n_labels = outputs.size()

        self._n_samples += batch_size * n_labels
        self._result += torch.sum(outputs != labels).item()

    @sync_all_reduce('_n_samples', '_result')
    def compute(self):
        return self._result / self._n_samples if self._n_samples > 0 else 0.0