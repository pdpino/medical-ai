import logging
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from pycocoevalcap.bleu import bleu_scorer

from medai.utils.nlp import PAD_IDX, indexes_to_strings

LOGGER = logging.getLogger(__name__)

class Bleu(Metric):
    """Computes BLEU metric up to N."""
    def __init__(self, n=4, pad_idx=PAD_IDX, output_transform=lambda x: x, device=None):
        if pad_idx != 0:
            # Otherwise, indexes_to_strings() function below gets ugly
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

            self.scorer += (generated, [gt])


    @sync_all_reduce('scorer')
    def compute(self):
        result = self.scorer.compute_score(verbose=0)

        # NOTE: there is a bug in pycocoevalcap, returning different types
        if isinstance(result, tuple):
            # Scores are first calculated: returned as tuple (scores, scores_by_instance)
            scores = result[0]
        elif isinstance(result, list):
            # Scores are cached, returned only scores
            scores = result
        else:
            # would be an internal error
            LOGGER.warning(
                'Warning: BleuScorer returned unknown type %s, %s',
                type(result), result,
            )
            scores = [0.0] * self._n

        return scores # np.mean(scores)
