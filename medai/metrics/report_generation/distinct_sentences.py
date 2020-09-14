from collections import Counter
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

from medai.utils.nlp import PAD_IDX, END_OF_SENTENCE_IDX

def _sentence_iterator(flat_report, end_idx=END_OF_SENTENCE_IDX):
    """Splits a flat_report into sentences, iterating on the fly.
    
    Args:
        flat_report: tensor of shape (n_words)

    Not very efficient!!
    """
    sentence_so_far = []
    for word in flat_report:
        word = word.item()
        sentence_so_far.append(word)
        if word == end_idx:
            yield sentence_so_far
            sentence_so_far = []
    
    if len(sentence_so_far) > 0:
        sentence_so_far.append(end_idx)
        yield sentence_so_far


class DistinctSentences(Metric):
    """Counts amount of different sentences generated."""
    @reinit__is_reduced
    def reset(self):
        self.sentences_seen = Counter()

        super().reset()

    @reinit__is_reduced
    def update(self, output):
        """Update on each step.

        output:
            reports_gen -- array of generated words,
                shape (batch_size, n_words)
            ground_truth -- unused
        """
        reports_gen, _ = output

        for report in reports_gen:
            for sentence in _sentence_iterator(report):
                sentence = ' '.join(str(word) for word in sentence)
                self.sentences_seen[sentence] += 1


    @sync_all_reduce('sentences_seen')
    def compute(self):
        return len(self.sentences_seen)
