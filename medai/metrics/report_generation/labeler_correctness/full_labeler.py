from abc import ABC, abstractmethod
import numpy as np

from medai.datasets.common import CHEXPERT_LABELS
from medai.metrics.report_generation import chexpert
from medai.metrics.report_generation.labeler_correctness.cache import ReportLabelsCache
from medai.utils.nlp import (
    ReportReader,
)
from medai.utils.timer import Timer


class FullLabeler(ABC):
    # TODO: reconcile LightLabeler and FullLabeler with the HolisticLabeler stuff from chexpert.py

    name = 'some-metric'
    diseases = ['dis1', 'dis2']

    no_finding_idx = None
    support_idxs = None

    def __init__(self, vocab):
        self._labels_by_report = ReportLabelsCache(self.name, 'reports', self.diseases)

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

    def _label_whole_reports(self, reports):
        """Split sentences, label them and apply union."""
        reports = [
            self._report_reader.idx_to_text(report)
            for report in reports
        ]

        # Check cache
        cache_miss = set(
            report
            for report in reports
            if report not in self._labels_by_report
        )

        if len(cache_miss) > 0:
            # Label reports not in cache (misses)
            cache_miss = list(cache_miss)
            with self.timer:
                new_labels = self._label_reports(cache_miss)

            # Insert new labels to cache
            self._labels_by_report.insert(cache_miss, new_labels)

        # Extract labels
        reports_labels = np.array([
            self._labels_by_report[report]
            for report in reports
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
            return self._label_whole_reports(reports)


class ChexpertFullLabeler(FullLabeler):
    # TODO: reuse this with ChexpertLightLabeler
    name = 'chexpert'
    diseases = CHEXPERT_LABELS

    def _label_reports(self, reports):
        labels = chexpert.apply_labeler_to_column(
            reports,
            fill_empty=-2, fill_uncertain=-1,
            quiet=True,
            caller_id='full-labeler',
        )

        return labels
