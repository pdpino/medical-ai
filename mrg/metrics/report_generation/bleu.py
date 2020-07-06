import numpy as np
import torch
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

from pycocoevalcap.bleu import bleu_scorer

from mrg.utils import PAD_IDX
from mrg.utils.nlp_metrics import indexes_to_strings

class Bleu(Metric):
    """Computes BLEU metric."""
    def __init__(self, n=4, pad_idx=PAD_IDX, output_transform=lambda x: x, device=None):
        if pad_idx != 0:
            # Otherwise, np.trim_zeros() function below gets ugly
            raise Exception('Bleu metric: pad idx must be 0!')

        self._n = n

        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self.scorer = bleu_scorer.BleuScorer(n=self._n)

        super().reset()

    @reinit__is_reduced
    def update(self, output):
        generated_words, seq = output
        # shape (both arrays): batch_size, sentence_len

        for generated, gt in zip(generated_words, seq):
            # shape (both): sentence_len

            generated, gt = indexes_to_strings(generated, gt)

            self.scorer += (generated, gt)


    @sync_all_reduce('scorer')
    def compute(self):
        scores, _ = self.scorer.compute_score(verbose=0)
        return np.mean(scores)