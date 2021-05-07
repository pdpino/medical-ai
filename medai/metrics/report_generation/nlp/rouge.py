from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

from pycocoevalcap.rouge import rouge

from medai.utils.nlp import indexes_to_strings

class RougeL(Metric):
    """Computes ROUGE-L metric."""
    def __init__(self, output_transform=lambda x: x, device=None):
        self.scorer = rouge.Rouge()

        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._n_samples = 0
        self._current_score = 0

        super().reset()

    @reinit__is_reduced
    def update(self, output):
        clean_reports_gen, clean_reports_gt = output
        # shape (both arrays): batch_size, sentence_len

        for generated, gt in zip(clean_reports_gen, clean_reports_gt):
            # shape (both): sentence_len

            generated, gt = indexes_to_strings(generated, gt)

            self._current_score += self.scorer.calc_score([generated], [gt])
            self._n_samples += 1

    @sync_all_reduce('_current_score', '_n_samples')
    def compute(self):
        return self._current_score / self._n_samples if self._n_samples > 0 else 0.0
