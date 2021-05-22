import re
import torch

from medai.datasets.common import CHEXPERT_DISEASES
from medai.utils.nlp import END_OF_SENTENCE_IDX, END_IDX

class WordMatcher:
    """Abstract class for matchers."""
    # TODO: use abc.abstractclass?
    def __init__(self, targets):
        self.state = 0

        self._targets = set(targets)

    def reset(self):
        self.state = 0

    def update(self, word):
        pass

    def _get_result(self):
        return self.state

    def close(self):
        state = self._get_result()
        self.reset()
        return state

    def __repr__(self):
        return self.__str__()

class AnyWordMatcher(WordMatcher):
    """If any word from the targets appear, returns True."""
    def update(self, word):
        if self.state == 0:
            self.state = int(word in self._targets)

    def __str__(self):
        return '|'.join(str(t) for t in self._targets)

class AllWordsMatcher(WordMatcher):
    """If all words from the target appear, returns True."""
    def update(self, word):
        self.state += int(word in self._targets)

    def _get_result(self):
        return int(self.state == len(self._targets))

    def __str__(self):
        return ' & '.join(str(t) for t in self._targets)

class MatcherGroup:
    """Contains multiple matchers."""
    def __init__(self, matchers):
        self.matchers = matchers

    def reset(self):
        for matcher in self.matchers:
            matcher.reset()

    def update(self, word):
        for matcher in self.matchers:
            matcher.update(word)

    def __str__(self):
        return list(self.matchers).__str__()

    def __repr__(self):
        return self.__str__()

class MatcherGroupAny(MatcherGroup):
    """If any from the group matches returns True."""
    def close(self):
        return sum(
            matcher.close()
            for matcher in self.matchers
        ) > 0

class MatcherGroupAll(MatcherGroup):
    """If all from the group match returns True."""
    def close(self):
        return sum(
            matcher.close()
            for matcher in self.matchers
        ) == len(self.matchers)

class Patterns:
    """Holds different patterns."""
    def __init__(self, *elements):
        self.elements = elements
    def __len__(self):
        return len(self.elements)
    def __iter__(self):
        yield from self.elements

class AllGroupsPattern(Patterns):
    pass

class AnyGroupPattern(Patterns):
    pass

class AllWordsPattern(Patterns):
    pass

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
_PATTERNS = {
    'neg': r'\b(no|without|free|not|removed|negative|clear|resolved)\b',
    'Enlarged Cardiomediastinum': AnyGroupPattern(
        AllGroupsPattern(r'cardiomediastinum|\bmediastinum|mediastinal', r'large|prominen|widen'),
        AllGroupsPattern('hilar', 'contour', r'large|prominen'),
    ),
    'Cardiomegaly': AnyGroupPattern(
        'cardiomegaly',
        AllGroupsPattern(
            r'heart', r'large|prominen|upper|top|widen',
        ),
        AllGroupsPattern(
            'cardiac', r'contour|silhouette', r'large|prominen|upper|top|widen',
        ),
    ),
    'Consolidation': r'consolidat',
    'Edema': AnyGroupPattern(
        r'edema|chf',
        AllWordsPattern('heart', 'failure'),
        AllGroupsPattern(r'pulmonar|vascular', r'congestion|prominence'),
    ),
    'Lung Lesion': AnyGroupPattern(
        r'mass|nodule|tumor|neoplasm|carcinoma|lump',
        AllGroupsPattern('nodular', r'densit|opaci[tf]'),
        AllWordsPattern('cavitary', 'lesion'),
    ),
    'Lung Opacity': AnyGroupPattern(
        r'opaci|infilitrate|infiltration|reticulation|scar',
        AllGroupsPattern(r'interstitial|reticular', r'marking|pattern|lung'),
        AllGroupsPattern(r'air[\s\-]*space', 'disease'),
        AllWordsPattern('parenchymal', 'scarring'),
        AllGroupsPattern(r'peribronchial|wall', 'thickening'),
    ),
    'Pneumonia': r'pneumonia|infectio',
    'Atelectasis': r'atelecta|collapse',
    'Pneumothorax': r'pneumothora',
    'Pleural Effusion': AnyGroupPattern(
        'effusion',
        AllGroupsPattern('pleural', r'fluid|effusion'),
    ),
    'Pleural Other': AnyGroupPattern(
        r'fibrosis|fibrothorax',
        AllGroupsPattern(r'pleural|pleuro\-(parenchymal|pericardial)', 'scar'),
        AllWordsPattern('pleural', 'thickening'),
    ),
    'Fracture': 'fracture',
    'Support Devices': AnyGroupPattern(
        # pylint: disable=line-too-long
        r'pacer|\bline\b|lines|picc|tube|valve|catheter|pacemaker|hardware|arthroplast|marker|icd|defib|device|drain\b|plate|screw|cannula|aparatus|coil|support|equipment|mediport',
    ),
}

def _resolve_regex_to_word_idxs(pattern, vocab):
    assert isinstance(pattern, str)
    re_pattern = re.compile(pattern)
    matched_words = [
        idx
        for word, idx in vocab.items()
        if re_pattern.search(word)
    ]

    return matched_words

def patterns_to_matcher(patterns, vocab):
    if isinstance(patterns, str):
        targets = _resolve_regex_to_word_idxs(patterns, vocab)
        return AnyWordMatcher(targets)

    if isinstance(patterns, AllWordsPattern):
        words = [vocab[word] for word in patterns]
        return AllWordsMatcher(words)

    if isinstance(patterns, AllGroupsPattern):
        return MatcherGroupAll([
            patterns_to_matcher(pattern, vocab)
            for pattern in patterns
        ])

    if isinstance(patterns, AnyGroupPattern):
        return MatcherGroupAny([
            patterns_to_matcher(pattern, vocab)
            for pattern in patterns
        ])

    raise Exception(f'Internal error: pattern type not understood {type(patterns)}')


class LighterLabeler:
    """Label reports using a Keyword approach, using patterns to detect negation."""
    name = 'some-name'
    diseases = ['dis1', 'dis2']

    no_finding_idx = None
    support_idxs = None

    use_numpy = False

    use_timer = False
    use_cache = False

    def __init__(self, vocab, device='cuda'):
        self._device = device

        self.negation_matcher = patterns_to_matcher(_PATTERNS['neg'], vocab)

        self.disease_matchers = [
            patterns_to_matcher(_PATTERNS[disease], vocab)
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
        reports_labels = []

        for report in reports:
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

            reports_labels.append(labels)

        reports_labels = torch.stack(reports_labels, dim=0)
        return reports_labels


class ChexpertLighterLabeler(LighterLabeler):
    name = 'chexpert'
    diseases = list(CHEXPERT_DISEASES[1:]) # Remove No Finding
