import torch
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class Specificity(Metric):
    """Computes specificity metric for a binary classification.
    
    Tested only for multilabel=False
    TODO: test with multilabel=True
    """
    @reinit__is_reduced
    def reset(self):
        self._pred_neg = 0
        self._total_neg = 0

        super().reset()

    @reinit__is_reduced
    def update(self, output):
        """Updates the metric state after a batch/epoch.
        
        output = (outputs, labels)
            - shape for both arrays: batch_size
            - in binary format
        """
        outputs, labels = output

        irrelevant_mask = (labels == 0)

        self._total_neg += torch.sum(irrelevant_mask).item()
        self._pred_neg += torch.sum(irrelevant_mask & (outputs == 0)).item()

    @sync_all_reduce('_total_neg', '_pred_neg')
    def compute(self):
        return self._pred_neg / self._total_neg if self._total_neg > 0 else 0.0