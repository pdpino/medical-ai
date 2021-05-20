from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

from medai.utils.nlp import sentence_iterator, ReportReader
from medai.datasets.common.sentences2organs.compute import get_main_organ_for_sentence

class OrganBySentence(Metric):
    """Checks if main-organ mentioned are correct in each sentence."""
    def __init__(self, vocab, **kwargs):
        super().__init__(**kwargs)

        self._reader = ReportReader(vocab)

    @reinit__is_reduced
    def reset(self):
        self._n_correct = 0
        self._n_total = 0

        super().reset()

    @reinit__is_reduced
    def update(self, output):
        """Update on each step.

        output:
            reports_gen -- list of lists of generated words,
                shape (batch_size, n_words_per_report)
            reports_gt -- list of lists of GT reports, same shape
        """
        reports_gen, reports_gt = output

        for report_gen, report_gt in zip(reports_gen, reports_gt):
            for sentence_gen, sentence_gt in zip(
                sentence_iterator(report_gen),
                sentence_iterator(report_gt),
                ):
                sentence_gen = self._reader.idx_to_text(sentence_gen)
                sentence_gt = self._reader.idx_to_text(sentence_gt)

                organ_gen = get_main_organ_for_sentence(sentence_gen)
                organ_gt = get_main_organ_for_sentence(sentence_gt)

                self._n_total += 1
                self._n_correct += int(organ_gen == organ_gt)

    @sync_all_reduce('_n_total', '_n_correct')
    def compute(self):
        return self._n_correct / self._n_total if self._n_total != 0 else 0
