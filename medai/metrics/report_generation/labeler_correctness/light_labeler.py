from abc import ABC, abstractmethod
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


class LightLabeler(ABC):
    name = 'some-metric'
    diseases = ['dis1', 'dis2']

    no_finding_idx = None
    support_idxs = None

    def __init__(self, vocab):
        self._labels_by_sentence = ReportLabelsCache(self.name, self.diseases)

        self._report_reader = ReportReader(vocab)

        self.global_timer = Timer()
        self.timer = Timer()

    @abstractmethod
    def _label_reports(self, reports):
        """Label a list of reports.

        Args:
            reports -- list of str, len = batch_size

        Returns:
            np.array of shape batch_size, n_diseases
        """

    def _reduce_report(self, report):
        """Reduces the labels of a report.

        Iterates over the sentences' labels, and apply a reduction to obtain
        an array of report labels.

        Args:
            report -- iterator of sentences' labels
        Returns:
            report labels, np.array of shape n_diseases
        """
        all_labels = np.array([self._labels_by_sentence[sentence] for sentence in report])
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
        # Version 1: TODO: which is faster? v1 or v2?
        # (v1 is not tested)
        # splitted_reports = [
        #     [
        #         self._report_reader.idx_to_text(sentence)
        #         for sentence in split_sentences_and_pad(report, pad=False)
        #     ]
        #     for report in reports
        # ]

        # Version 2
        splitted_reports = [
            [
                self._report_reader.idx_to_text(sentence)
                for sentence in sentence_iterator(report)
            ]
            for report in reports
        ]
        # list of lists of str, shape: batch_size, n_sentences

        # Check cache
        cache_miss = set(
            sentence
            for report in splitted_reports
            for sentence in report
            if sentence not in self._labels_by_sentence
        )

        if len(cache_miss) > 0:
            # Label reports not in cache (misses)
            cache_miss = list(cache_miss)
            with self.timer:
                new_labels = self._label_reports(cache_miss)

            # Insert new labels to cache
            self._labels_by_sentence.insert(cache_miss, new_labels)


        # Apply union over sentences
        # Union is applied as max over NAN (-2), UNC (-1), NEG (0) and POS (1)
        # If one sentence contains a positive mention --> whole report is positive
        # and so on
        # FIXME: for chexpert, the max function is not appropriate for "No Finding"
        reports_labels = np.array([
            self._reduce_report(report)
            for report in splitted_reports
        ])
        # np.array shape: batch_size, n_diseases

        return reports_labels


    def __call__(self, reports):
        """Labels a batch of generated and ground_truth reports.

        Args:
            reports -- tensor of shape batch_size, n_words
        Returns:
            labels, tensor of shape batch_size, n_labels
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
        )

        return labels
