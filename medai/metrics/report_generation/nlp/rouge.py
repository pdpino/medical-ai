from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

from pycocoevalcap.rouge import rouge

from medai.utils.nlp import PAD_IDX, indexes_to_strings


class RougeL(Metric):
    """Computes ROUGE-L metric."""
    def __init__(self, pad_idx=PAD_IDX, output_transform=lambda x: x, device=None):
        if pad_idx != 0:
            raise Exception('ROUGE-L metric: pad idx must be 0!')

        self.scorer = rouge.Rouge()
        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._n_samples = 0
        self._current_score = 0

        super().reset()

    @reinit__is_reduced
    def update(self, output):
        generated_words, seq = output
        # shape (both arrays): batch_size, sentence_len

        for generated, gt in zip(generated_words, seq):
            # shape (both): sentence_len

            generated, gt = indexes_to_strings(generated, gt)

            self._current_score += self.scorer.calc_score([generated], [gt])
            self._n_samples += 1

    @sync_all_reduce('_current_score', '_n_samples')
    def compute(self):
        return self._current_score / self._n_samples if self._n_samples > 0 else 0.0
