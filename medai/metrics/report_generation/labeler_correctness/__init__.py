import numpy as np
from ignite.metrics import Metric, MetricUsage
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

from medai.utils import divide_arrays


def _replace_nan_and_uncertain_(arr, nan_with=0, uncertain_with=1):
    """Replaces -2 and -1 values in an array inplace."""
    _NAN = -2
    _UNC = -1

    arr[arr == _NAN] = nan_with
    arr[arr == _UNC] = uncertain_with


class MedicalLabelerCorrectness(Metric):
    METRICS = ['acc', 'prec', 'recall', 'spec', 'npv', 'f1']

    def __init__(self, labeler, output_transform=lambda x: x, device=None):
        self.labeler = labeler
        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._tp = 0
        self._fp = 0
        self._tn = 0
        self._fn = 0

        super().reset()

    @reinit__is_reduced
    def update(self, output):
        generated, gt = output
        # shape (both arrays): batch_size, n_words

        generated_labels = self.labeler(generated)
        gt_labels = self.labeler(gt)
        # shape (both): batch_size, n_labels

        _replace_nan_and_uncertain_(generated_labels)
        _replace_nan_and_uncertain_(gt_labels)

        self._tp += np.sum((generated_labels == 1) & (gt_labels == 1), axis=0)
        self._fp += np.sum((generated_labels == 1) & (gt_labels == 0), axis=0)
        self._tn += np.sum((generated_labels == 0) & (gt_labels == 0), axis=0)
        self._fn += np.sum((generated_labels == 0) & (gt_labels == 1), axis=0)
        # shape (all): n_labels

    @sync_all_reduce('_tp', '_fp', '_tn', '_fn')
    def compute(self):
        total = self._tp + self._fp + self._tn + self._fn

        accuracy = divide_arrays(self._tp + self._tn, total)
        precision = divide_arrays(self._tp, self._tp + self._fp)
        recall = divide_arrays(self._tp, self._tp + self._fn)
        specificity = divide_arrays(self._tn, self._tn + self._fp)
        npv = divide_arrays(self._tn, self._tn + self._fn)
        f1 = divide_arrays(precision * recall * 2, precision + recall)
        # shape (all): n_diseases

        return {
            'acc': accuracy,
            'prec': precision,
            'recall': recall,
            'spec': specificity,
            'npv': npv,
            'f1': f1,
        }