"""Define matchers and patterns."""


### Words matchers
# TODO: use abc.abstractclass?
class WordMatcher:
    """Abstract class for matchers."""
    start_state = -2

    def __init__(self, targets):
        self._state = None

        self._targets = set(targets)

    def reset(self):
        self._state = self.start_state

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

    def __str__(self):
        return list(self._matchers).__str__()

    def __repr__(self):
        return self.__str__()

class MatcherGroupAny(MatcherGroup):
    """If any from the group matches returns True."""
    def close(self):
        any_is_1 = False
        for matcher in self._matchers:
            result = matcher.close() # all matchers must be closed!
            any_is_1 = any_is_1 or result == 1
        return 1 if any_is_1 else -2

class MatcherGroupAll(MatcherGroup):
    """If all from the group match returns True."""
    def close(self):
        all_are_1 = True
        for matcher in self._matchers:
            result = matcher.close() # all matchers must be closed!
            all_are_1 = all_are_1 and result == 1
        return 1 if all_are_1 else -2


### Patterns (use in static declarations)
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
