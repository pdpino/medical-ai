import logging
import re
import torch

from medai.utils.nlp import END_OF_SENTENCE_IDX, END_IDX, END_OF_SENTENCE_TOKEN, END_TOKEN
from medai.metrics.report_generation.abn_match import matchers

LOGGER = logging.getLogger(__name__)

# TODO: design for unmentions? (i.e. False positives)
# REVIEW: ideas to improve negation handling:
# Idea 1: instead of using a global negative-matching, use one per matcher,
# and only activates if is given before a target word.
# For example:
#  "no cardiomegaly" --> cardiomegaly == 0 (appears, but there is negation before target word)
#  "cardiomegaly, but no ..." --> cardiomegaly == 1 (negation is after, so it does not count)
#
# Idea 2: use the global negation, but block calls to matcher.update()
# if there is a negation already. This may make things even faster!


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



_NEG_PATTERN = r'\b(no|without|free|not|removed|negative|clear|resolved)\b'


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

    use_numpy = False

    use_timer = False
    use_cache = False

    def __init__(self, vocab, use_idx=True, device='cuda'):
        self._device = device

        self.__check_patterns_valid()

        self.punctuation_handler = PunctuationHandler(use_idx)

        self.negation_matcher = _patterns_to_matcher(_NEG_PATTERN, vocab, use_idx)

        self.disease_matchers = [
            _patterns_to_matcher(self.patterns[disease], vocab, use_idx)
            for disease in self.diseases
        ]

    def __check_patterns_valid(self):
        for value in self.patterns.values():
            assert isinstance(value, (str, matchers.Patterns))

    def __init_array(self, array):
        # pylint: disable=not-callable
        return torch.tensor(
            array,
            dtype=torch.int8,
            device=self._device,
        )

    def _reset_matchers(self):
        self.negation_matcher.reset()

        for matcher in self.disease_matchers:
            matcher.reset()

    def _update_matchers(self, word):
        self.negation_matcher.update(word)

        for matcher in self.disease_matchers:
            matcher.update(word)

    def _close_matchers(self):
        negate = int(self.negation_matcher.close() == 1)

        labels = []
        for matcher in self.disease_matchers:
            result = matcher.close() # All matchers must be closed
            if negate and result == 1:
                result = 0

            labels.append(result)
        return self.__init_array(labels)

    def label_report(self, report):
        if isinstance(report, str):
            report = report.split()

        labels = self.__init_array([-2] * len(self.diseases))

        self._reset_matchers()

        is_closed = False
        for word in report:
            if self.punctuation_handler.is_end(word):
                break

            if self.punctuation_handler.is_end_of_sentence(word):
                labels = torch.maximum(self._close_matchers(), labels)
                is_closed = True
            else:
                self._update_matchers(word)
                is_closed = False

        if not is_closed:
            labels = torch.maximum(self._close_matchers(), labels)

        labels[labels == -2] = 0

        return labels

    def label_reports(self, reports):
        assert isinstance(reports, list), f'reports must be a list, got {type(reports)}'

        return torch.stack([
            self.label_report(report)
            for report in reports
        ], dim=0)
        # shape: n_reports, n_labels

    def __call__(self, reports):
        return self.label_reports(reports)
