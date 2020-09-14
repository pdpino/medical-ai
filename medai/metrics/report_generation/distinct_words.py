from collections import Counter
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

from medai.utils.nlp import PAD_IDX

class DistinctWords(Metric):
    """Counts amount of different words generated."""
    def __init__(self, ignore_pad=True, output_transform=lambda x: x, device=None):
        self.ignore_pad = ignore_pad

        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self.words_seen = Counter()

        super().reset()

    @reinit__is_reduced
    def update(self, output):
        """Update on each step.
        
        output:
            generated_words -- array of generated words,
                shape (batch_size, *)
            ground_truth -- unused
        """
        generated_words, _ = output

        for word in generated_words.view(-1):
            word = int(word.item())
            if self.ignore_pad and word == PAD_IDX:
                continue
            self.words_seen[word] += 1

    @sync_all_reduce('words_seen')
    def compute(self):
        return len(self.words_seen)
