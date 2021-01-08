from abc import ABC, abstractmethod
from functools import reduce
import os
import pandas as pd
import numpy as np

from medai.datasets.common import CHEXPERT_LABELS
from medai.metrics.report_generation import chexpert
from medai.metrics.report_generation.labeler_correctness.cache import SentencesLabelCache
from medai.utils.nlp import (
    ReportReader,
    sentence_iterator,
)
from medai.utils.timer import Timer


class LightLabeler(ABC):
    name = 'some-metric'
    diseases = ['dis1', 'dis2']

    def __init__(self, vocab):
        self._labels_by_sentence = SentencesLabelCache(self.name, self.diseases)

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
        pass


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
        np_max = np.vectorize(max)
        all_nan_labels = np.full(len(self.diseases), -2)
        reports_labels = np.array([
            reduce(
                np_max,
                [self._labels_by_sentence[sentence] for sentence in report],
                all_nan_labels,
            )
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

    def _label_reports(self, reports):
        _column_name = 'sentences'

        reports_df = pd.DataFrame(reports, columns=[_column_name])
        labels = chexpert.apply_labeler_to_column(
            reports_df, _column_name,
            fill_empty=-2, fill_uncertain=-1,
            quiet=True,
        )

        return labels
