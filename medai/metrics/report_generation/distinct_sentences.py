from collections import Counter
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

from medai.utils.nlp import sentence_iterator


class DistinctSentences(Metric):
    """Counts amount of different sentences generated.

    Note:
        The use of sentence_iterator() is not very efficient for the hierarchical-decoder:
            - sentences were already separated in the model output
            - sentences were flattened into a flat report, to comply with flat-decoder
            - the flat report is split with sentence_iterator() again
    """
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
            for sentence in sentence_iterator(report):
                sentence = ' '.join(str(word) for word in sentence)
                self.sentences_seen[sentence] += 1


    @sync_all_reduce('sentences_seen')
    def compute(self):
        return len(self.sentences_seen)
