import logging
import re
import torch

from medai.utils.nlp import END_OF_SENTENCE_IDX, END_IDX
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


def _resolve_regex_to_word_idxs(pattern, vocab):
    assert isinstance(pattern, str)
    re_pattern = re.compile(pattern)
    matched_words = [
        idx
        for word, idx in vocab.items()
        if re_pattern.search(word)
    ]

    return matched_words

def _patterns_to_matcher(patterns, vocab):
    if isinstance(patterns, str):
        targets = _resolve_regex_to_word_idxs(patterns, vocab)
        return matchers.AnyWordMatcher(targets)

    if isinstance(patterns, matchers.AllWordsPattern):
        words_idx = [vocab.get(word, -1) for word in patterns]

        absent_words = [
            word
            for word, idx in zip(patterns, words_idx)
            if idx == -1
        ]
        if absent_words:
            LOGGER.warning('AllWordsMatcher: exact-words not found in vocab: %s', absent_words)
        return matchers.AllWordsMatcher(words_idx)

    if isinstance(patterns, matchers.AllGroupsPattern):
        return matchers.MatcherGroupAll([
            _patterns_to_matcher(pattern, vocab)
            for pattern in patterns
        ])

    if isinstance(patterns, matchers.AnyGroupPattern):
        return matchers.MatcherGroupAny([
            _patterns_to_matcher(pattern, vocab)
            for pattern in patterns
        ])

    raise Exception(f'Internal error: pattern type not understood {type(patterns)}')


class AbnormalityLabeler:
    """Label reports using a Keyword approach, using patterns to detect negation."""
    name = 'some-name'
    diseases = ['dis1', 'dis2']
    patterns = {}

    no_finding_idx = None
    support_idxs = None

    use_numpy = False

    use_timer = False
    use_cache = False

    def __init__(self, vocab, device='cuda'):
        self._device = device

        self.negation_matcher = _patterns_to_matcher(self.patterns['neg'], vocab)

        self.disease_matchers = [
            _patterns_to_matcher(self.patterns[disease], vocab)
            for disease in self.diseases
        ]

    def _reset_matchers(self):
        self.negation_matcher.reset()

        for matcher in self.disease_matchers:
            matcher.reset()

    def _update_matchers(self, word):
        self.negation_matcher.update(word)

        for matcher in self.disease_matchers:
            matcher.update(word)

    def _close_matchers(self):
        negation = self.negation_matcher.close()
        forced_zero = 1 - negation

        # pylint: disable=not-callable
        return torch.tensor([
            min(matcher.close(), forced_zero)
            for matcher in self.disease_matchers
        ], dtype=torch.uint8, device=self._device)

    def __call__(self, reports):
        return self.label_reports(reports)

    def label_reports(self, reports):
        return torch.stack([
            self.label_report(report)
            for report in reports
        ], dim=0)
        # shape: n_reports, n_labels

    def label_report(self, report):
        labels = torch.zeros(len(self.diseases), dtype=torch.uint8, device=self._device)

        self._reset_matchers()

        is_closed = False
        for word in report:
            if word == END_IDX:
                break

            if word == END_OF_SENTENCE_IDX:
                labels = torch.maximum(self._close_matchers(), labels)
                is_closed = True
            else:
                self._update_matchers(word)
                is_closed = False

        if not is_closed:
            labels = torch.maximum(self._close_matchers(), labels)

        return labels
