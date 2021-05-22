from abc import ABC, abstractmethod
import logging
import pandas as pd
import numpy as np

from medai.datasets.common import CHEXPERT_LABELS
from medai.metrics.report_generation import chexpert
from medai.metrics.report_generation.labeler_correctness.cache import ReportLabelsCache
from medai.utils.nlp import (
    ReportReader,
    sentence_iterator,
)
from medai.utils.timer import Timer

LOGGER = logging.getLogger(__name__)

class CacheHitCounter:
    def __init__(self):
        self.reset()

    def reset(self):
        self._misses = 0
        self._unique_misses = 0
        self._total = 0

    def update(self, n_misses, n_unique_misses, n_total):
        self._misses += n_misses
        self._unique_misses += n_unique_misses
        self._total += n_total

    def compute(self):
        return self._misses, self._unique_misses, self._total


_IGNORE_TOKENS = ('END', 'PAD', '.', 'xxxx', '&lt', '/', '(', ')', 'UNK', '-') # ','
def clean_sentence(sentence):
    sentence = [
        token
        for token in sentence
        if token not in _IGNORE_TOKENS
    ]
    # Remove consecutive equal tokens
    return [
        token
        for i, token in enumerate(sentence)
        if i == 0 or token != sentence[i-1]
    ]


def _replace_nan_and_uncertain_(arr, nan_with=0, uncertain_with=1):
    """Replaces -2 and -1 values in an array inplace."""
    _NAN = -2
    _UNC = -1

    arr[arr == _NAN] = nan_with
    arr[arr == _UNC] = uncertain_with


class LightLabeler(ABC):
    name = 'some-metric'
    diseases = ['dis1', 'dis2']

    no_finding_idx = None
    support_idxs = None

    use_numpy = True

    use_timer = True
    use_cache = True

    def __init__(self, vocab):
        self._labels_by_sentence = ReportLabelsCache(self.name, 'sentences', self.diseases)

        self._report_reader = ReportReader(vocab, ignore_pad=True)

        self.global_timer = Timer()
        self.timer = Timer()

        # FIXME: this is not implemented in full_labeler, fix code duplication!
        self.hit_counter = CacheHitCounter()

    @abstractmethod
    def _label_reports(self, reports):
        """Label a list of reports.

        Args:
            reports -- list of str, len = batch_size

        Returns:
            np.array of shape batch_size, n_diseases
        """

    def _reduce_sentences(self, report):
        """Reduces the labels of a report, given as labels by sentences.

        Iterates over the sentences' labels, and apply a reduction to obtain
        an array of report labels.

        Reduction is applied as max over NAN (-2), UNC (-1), NEG (0) and POS (1).
        If one sentence contains a positive mention --> whole report is positive.

        If self.no_finding_idx is defined, it indicates the index of a disease
        that has to be treated different (as it represents absence of other diseases).

        Args:
            report -- iterator of sentences' labels
        Returns:
            report labels, np.array of shape n_diseases
        """
        if not report:
            LOGGER.debug('Found empty report')
            return np.zeros(len(self.diseases))

        all_labels = np.array([
            self._labels_by_sentence[sentence]
            for sentence in report
        ])
        # shape: n_sentences, n_diseases

        reduced_labels = all_labels.max(axis=0)
        # shape: n_diseases

        if self.no_finding_idx is None:
            return reduced_labels

        # The no-finding disease needs to be treated different.
        # If any other disease is present --> no-finding == 0
        # If all diseases are absent --> no-finding == 1

        # Consider only actual diseases, by masking no-finding and support-devices
        ignore_mask = np.zeros(len(self.diseases))
        ignore_mask[self.no_finding_idx] = 1
        if self.support_idxs is not None:
            ignore_mask[self.support_idxs] = 1

        # Filter only actual diseases
        disease_labels = np.ma.array(reduced_labels, mask=ignore_mask)

        # Check if all diseases are absent
        no_findings = np.ma.all((disease_labels == 0) | (disease_labels == -2))

        # Override reduced_labels array
        reduced_labels[self.no_finding_idx] = no_findings

        return reduced_labels

    def _split_sentences_and_label(self, reports):
        """Split sentences, label them and apply union."""
        # Split reports into sentences
        splitted_reports = [
            [
                self._report_reader.idx_to_text(clean_sentence(sentence))
                for sentence in sentence_iterator(report)
            ]
            for report in reports
        ]
        # list of lists of str, shape: batch_size, n_sentences

        n_sentences = sum(len(r) for r in splitted_reports)

        # Check cache
        cache_miss = [
            sentence
            for report in splitted_reports
            for sentence in report
            if sentence not in self._labels_by_sentence
        ]
        n_sentences_missed = len(cache_miss)

        # Remove repeated sentences
        cache_miss = set(cache_miss)
        n_unique_missed = len(cache_miss)

        # Update hit-miss counter
        self.hit_counter.update(n_sentences_missed, n_unique_missed, n_sentences)

        if len(cache_miss) > 0:
            # Label reports not in cache (misses)

            cache_miss = list(cache_miss)
            with self.timer:
                new_labels = self._label_reports(cache_miss)

                assert len(new_labels) == len(cache_miss), (
                    f'_label_reports() output mismatch, {new_labels.shape} vs {len(cache_miss)}',
                )

            # Insert new labels to cache
            self._labels_by_sentence.insert(cache_miss, new_labels)

        reports_labels = np.array([
            self._reduce_sentences(report) # shape: n_diseases
            for report in splitted_reports
        ])
        # np.array shape: batch_size, n_diseases

        _replace_nan_and_uncertain_(reports_labels)

        return reports_labels

    def __call__(self, reports):
        """Labels a batch of generated and ground_truth reports.

        Args:
            reports -- list of lists of shape batch_size, n_words
        Returns:
            labels, np.array of shape batch_size, n_labels
        """
        with self.global_timer:
            return self._split_sentences_and_label(reports)


class ChexpertLightLabeler(LightLabeler):
    name = 'chexpert'
    diseases = CHEXPERT_LABELS

    no_finding_idx = 0
    support_idxs = 13

    def _label_reports(self, reports):
        _column_name = 'sentences'

        reports_df = pd.DataFrame(reports, columns=[_column_name])
        labels = chexpert.apply_labeler_to_column(
            reports_df, _column_name,
            fill_empty=-2, fill_uncertain=-1,
            quiet=True,
            caller_id='light-labeler',
        )

        return labels

class DummyLabeler(LightLabeler):
    """Use it for debugging!!."""
    name = 'dummy'
    diseases = CHEXPERT_LABELS

    no_finding_idx = 0
    support_idxs = 13

    def _label_reports(self, reports):
        return np.zeros((len(reports), len(self.diseases)))
