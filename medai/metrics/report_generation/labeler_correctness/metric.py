# import torch
# import numpy as np
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

from medai.utils import divide_arrays, divide_tensors


class MedicalLabelerCorrectness(Metric):
    METRICS = ['acc', 'prec', 'recall', 'spec', 'npv', 'f1']

    def __init__(self, labeler, output_transform=lambda x: x, device=None):
        super().__init__(output_transform=output_transform, device=device)
        self.labeler = labeler

        if not hasattr(labeler, 'use_numpy'):
            raise Exception(f'Internal error: set use_numpy in {labeler.__class__.__name__}')

        if labeler.use_numpy:
            self._divide_results = divide_arrays
        else:
            # use torch tensors
            self._divide_results = divide_tensors

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

        self._tp += ((generated_labels == 1) & (gt_labels == 1)).sum(0)
        self._fp += ((generated_labels == 1) & (gt_labels == 0)).sum(0)
        self._tn += ((generated_labels == 0) & (gt_labels == 0)).sum(0)
        self._fn += ((generated_labels == 0) & (gt_labels == 1)).sum(0)
        # shape (all): n_labels

    @sync_all_reduce('_tp', '_fp', '_tn', '_fn')
    def compute(self):
        total = self._tp + self._fp + self._tn + self._fn

        accuracy = self._divide_results(self._tp + self._tn, total)
        precision = self._divide_results(self._tp, self._tp + self._fp)
        recall = self._divide_results(self._tp, self._tp + self._fn)
        specificity = self._divide_results(self._tn, self._tn + self._fp)
        npv = self._divide_results(self._tn, self._tn + self._fn)
        f1 = self._divide_results(precision * recall * 2, precision + recall)
        # shape (all): n_diseases

        return {
            'acc': accuracy,
            'prec': precision,
            'recall': recall,
            'spec': specificity,
            'npv': npv,
            'f1': f1,
        }
