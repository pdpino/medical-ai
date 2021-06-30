"""Define matchers and patterns."""

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
