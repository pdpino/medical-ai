"""Define matchers and patterns."""


### Words matchers
# TODO: use abc.abstractclass?
class WordMatcher:
    """Abstract class for matchers."""
    def __init__(self, targets):
        self._state = None

        self._targets = set(targets)

    def reset(self):
        self._state = -2

    def update(self, word):
        pass

    def _state_to_label(self):
        return self._state

    def close(self):
        label = self._state_to_label()
        self.reset()
        return label

    def __repr__(self):
        return self.__str__()

class AnyWordMatcher(WordMatcher):
    """If any word from the targets appear, returns True."""
    def update(self, word):
        if self._state == -2:
            if word in self._targets:
                self._state = 1

    def __str__(self):
        return '|'.join(str(t) for t in self._targets)

class AllWordsMatcher(WordMatcher):
    """If all words from the target appear, returns True."""
    def reset(self):
        self._state = set(self._targets)

    def update(self, word):
        if word in self._state:
            self._state.remove(word)

    def _state_to_label(self):
        if len(self._state) == 0:
            return 1
        return -2

    def __str__(self):
        return ' & '.join(str(t) for t in self._targets)


### Group matchers
class MatcherGroup:
    """Contains multiple matchers."""
    def __init__(self, matchers):
        self._matchers = matchers

    def reset(self):
        for matcher in self._matchers:
            matcher.reset()

    def update(self, word):
        for matcher in self._matchers:
            matcher.update(word)

    def close(self):
        pass

    def __str__(self):
        return list(self._matchers).__str__()

    def __repr__(self):
        return self.__str__()

class MatcherGroupAny(MatcherGroup):
    """If any from the group matches returns True."""
    def close(self):
        return max(
            matcher.close()
            for matcher in self._matchers
        )

class MatcherGroupAll(MatcherGroup):
    """If all from the group match returns True."""
    def close(self):
        all_are_1 = True
        for matcher in self._matchers:
            result = matcher.close() # all matchers must be closed!
            all_are_1 = all_are_1 and result == 1
        return 1 if all_are_1 else -2


### Body-part matcher
class BodyPartStatusMatcher(MatcherGroup):
    """Applies a <body part> <status> pattern."""
    def __init__(self, body_part_matcher, normal_matcher, abnormal_matcher):
        super().__init__([body_part_matcher, normal_matcher, abnormal_matcher])

    def close(self):
        body_part_matcher, normal_matcher, abnormal_matcher = self._matchers

        body_part = body_part_matcher.close()
        normal = normal_matcher.close()
        abnormal = abnormal_matcher.close()

        if body_part != 1:
            # Did not mention body part
            return -2

        if abnormal == 1:
            # Mentioned body part and abnormal words
            return 1

        if normal == 1:
            # Mentioned body part and normal words
            return 0

        # Mention body part but no normal/abnormal words
        # --> probably a false positive
        return -2


### Patterns (use in static declarations)
class Patterns:
    """Holds different patterns."""
    def __init__(self, *elements):
        self.elements = list(elements)
    def __len__(self):
        return len(self.elements)
    def __iter__(self):
        yield from self.elements
    def __repr__(self):
        return self.__str__()

class AllGroupsPattern(Patterns):
    def __str__(self):
        return ' & '.join(f"({str(e)})" for e in self.elements)

class AnyGroupPattern(Patterns):
    def __str__(self):
        return ' | '.join(f"({str(e)})" for e in self.elements)

class AllWordsPattern(Patterns):
    def __str__(self):
        s = 'Exact: '
        s += ' & '.join(str(e) for e in self.elements)
        return s


class BodyPartStatusPattern(Patterns):
    def __init__(self, *, body=None, normal=None, abnormal=None):
        super().__init__(body, normal, abnormal)

        assert body is not None
        assert normal is not None
        assert abnormal is not None

    def __str__(self):
        assert len(self.elements) == 3
        # pylint: disable=unbalanced-tuple-unpacking
        e1, e2, e3 = self.elements
        return f'body: {e1}, normal: {e2}, abnormal: {e3}'
