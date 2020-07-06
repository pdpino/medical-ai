import torch
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

from pycocoevalcap.cider import cider_scorer

from mrg.utils import PAD_IDX
from mrg.utils.nlp_metrics import indexes_to_strings


class CiderD(Metric):
    """Computes Cider-D metric."""
    def __init__(self, n=4, pad_idx=PAD_IDX, output_transform=lambda x: x, device=None):
        if pad_idx != 0:
            raise Exception('Cider-D metric: pad idx must be 0!')

        self._n = n

        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self.scorer = cider_scorer.CiderScorer(n=self._n)

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
        score, _ = self.scorer.compute_score()
        return score