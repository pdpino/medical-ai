# from collections import defaultdict
import logging
import re
import torch

from medai.utils.nlp import END_OF_SENTENCE_IDX, END_IDX, END_OF_SENTENCE_TOKEN, END_TOKEN
from medai.metrics.report_generation.abn_match import (
    matchers,
    uncertainty,
    # collectors,
)

LOGGER = logging.getLogger(__name__)


def _resolve_regex_to_word_idxs(pattern, vocab, use_idx=True):
    assert isinstance(pattern, str)
    re_pattern = re.compile(pattern)
    matched_words = [
        idx if use_idx else word
        for word, idx in vocab.items()
        if re_pattern.search(word)
    ]

    return matched_words

def _patterns_to_matcher(patterns, vocab, use_idx=True):
    if isinstance(patterns, str):
        targets = _resolve_regex_to_word_idxs(patterns, vocab, use_idx=use_idx)
        if len(targets) == 0:
            LOGGER.warning('Pattern matched no words from the vocab: %s', patterns)
        return matchers.AnyWordMatcher(targets)

    if isinstance(patterns, matchers.WordCollectorPattern):
        targets = []
        for pattern in patterns:
            targets.extend(_resolve_regex_to_word_idxs(pattern, vocab, use_idx=use_idx))
        return matchers.WordCollector(targets)

    if isinstance(patterns, matchers.AllWordsPattern):
        targets = []
        absent_words = []
        for word in patterns:
            idx = vocab.get(word, -1)
            if idx == -1:
                absent_words.append(word)
            targets.append(idx if use_idx else word)

        if absent_words:
            LOGGER.warning('AllWordsMatcher: exact-words not found in vocab: %s', absent_words)
        return matchers.AllWordsMatcher(targets)

    if isinstance(patterns, matchers.BodyPartStatusPattern):
        assert len(patterns) == 3

        return matchers.BodyPartStatusMatcher(*[
            _patterns_to_matcher(pattern, vocab, use_idx)
            for pattern in patterns
        ])


    if isinstance(patterns, matchers.AllGroupsPattern):
        MatcherClass = matchers.MatcherGroupAll
    elif isinstance(patterns, matchers.AnyGroupPattern):
        MatcherClass = matchers.MatcherGroupAny
    else:
        raise Exception(f'Internal error: pattern type not understood {type(patterns)}')

    return MatcherClass([
        _patterns_to_matcher(pattern, vocab, use_idx)
        for pattern in patterns
    ])



# pylint: disable=line-too-long
_NEG_PATTERN = r'\b(no|without|free|not|removed|removal|negative|clear|cleared|resolved|resolution)\b'

class PunctuationHandler:
    def __init__(self, use_idx=True):
        if use_idx:
            self._is_end = lambda x: x == END_IDX
            self._is_end_of_sentence = lambda x: x == END_OF_SENTENCE_IDX
        else:
            self._is_end = lambda x: x == END_TOKEN
            self._is_end_of_sentence = lambda x: x == END_OF_SENTENCE_TOKEN

    def is_end(self, token):
        return self._is_end(token)

    def is_end_of_sentence(self, token):
        return self._is_end_of_sentence(token)


class AbnormalityLabeler:
    """Label reports using a Keyword approach, using patterns to detect negation."""
    name = 'some-name'
    metric_name = 'some-metric-name'
    diseases = ['dis1', 'dis2']
    patterns = {}

    no_finding_idx = None
    support_idxs = None
    lung_diseases = []

    use_numpy = False

    use_timer = False
    use_cache = False

    def __init__(self, vocab, use_idx=True, device='cuda'):
        self._device = device

        self.__check_patterns_valid()
        self.__check_diseases_valid()

        self.punctuation_handler = PunctuationHandler(use_idx)

        self.negation_matcher = _patterns_to_matcher(_NEG_PATTERN, vocab, use_idx)
        self.uncertainty_matcher = _patterns_to_matcher(uncertainty.UNC_PATTERN, vocab, use_idx)

        # self.lung_location_collector = _patterns_to_matcher(
        #     collectors.LUNG_LOCATIONS, vocab, use_idx,
        # )

        self.disease_matchers = [
            _patterns_to_matcher(self.patterns[disease], vocab, use_idx)
            for disease in self.diseases
        ]

    def __check_patterns_valid(self):
        for value in self.patterns.values():
            assert isinstance(value, (str, matchers.Patterns))

    def __check_diseases_valid(self):
        assert isinstance(self.lung_diseases, (tuple, list))

    def __init_array(self, array):
        # pylint: disable=not-callable
        return torch.tensor(
            array,
            dtype=torch.int8,
            device=self._device,
        )

    def __iter_lung_diseases(self):
        for index in self.lung_diseases:
            yield self.diseases[index]

    def _reset_matchers(self):
        self.negation_matcher.reset()
        self.uncertainty_matcher.reset()
        # self.lung_location_collector.reset()

        for matcher in self.disease_matchers:
            matcher.reset()

    def _update_matchers(self, word):
        self.negation_matcher.update(word)
        self.uncertainty_matcher.update(word)
        # self.lung_location_collector.update(word)

        for matcher in self.disease_matchers:
            matcher.update(word)

    def _close_matchers(self):
        negate = int(self.negation_matcher.close() == 1)
        uncertain = int(self.uncertainty_matcher.close() == 1)
        # lung_locations = self.lung_location_collector.close()

        labels = []
        for matcher in self.disease_matchers:
            result = matcher.close() # All matchers must be closed
            if result == 1:
                if negate:
                    result = 0
                elif uncertain:
                    result = -1

            labels.append(result)
        return self.__init_array(labels) # , lung_locations

    def label_report(self, report):
        if isinstance(report, str):
            report = report.split()

        labels = self.__init_array([-2] * len(self.diseases))
        # lung_locations = defaultdict(list)

        self._reset_matchers()

        is_closed = False
        for word in report:
            # TODO: add warning if is a word, and use_idx was True
            # and viceversa

            if self.punctuation_handler.is_end(word):
                break

            if self.punctuation_handler.is_end_of_sentence(word):
                new_labels = self._close_matchers() # new_locations
                labels = torch.maximum(new_labels, labels)
                # if new_locations:
                #     for disease, value in zip(self.__iter_lung_diseases(), labels):
                #         if value == 1 or value == -1:
                #             lung_locations[disease].append(new_locations)

                is_closed = True
            else:
                self._update_matchers(word)
                is_closed = False

        if not is_closed:
            new_labels = self._close_matchers() # new_locations
            labels = torch.maximum(new_labels, labels)
            # if new_locations:
            #     for disease, value in zip(self.__iter_lung_diseases(), labels):
            #         if value == 1 or value == -1:
            #             lung_locations[disease].append(new_locations)

        labels[labels == -2] = 0

        return labels #, lung_locations

    def label_reports(self, reports):
        assert isinstance(reports, list), f'reports must be a list, got {type(reports)}'

        # FIXME: it wont work with lung_locations!!
        return torch.stack([
            self.label_report(report)
            for report in reports
        ], dim=0)
        # shape: n_reports, n_labels

    def __call__(self, reports):
        return self.label_reports(reports)
