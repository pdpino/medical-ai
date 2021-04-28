import random
from collections import Counter, defaultdict
import torch
from torch import nn
from torch.nn.functional import one_hot


def _get_stats(dataset):
    word_appearances = defaultdict(lambda: 0)
    reports_lens = Counter()

    for report in dataset.iter_reports_only():
        report = report['tokens_idxs']
        for word in report:
            word_appearances[word] += 1
        reports_lens[len(report)] += 1

    word_appearances = sorted(list(word_appearances.items()),
                              key=lambda x: x[1],
                              reverse=True)

    return word_appearances, reports_lens


class MostCommonWords(nn.Module):
    def __init__(self, dataset, k_first=100):
        super().__init__()

        self.vocab_size = len(dataset.get_vocab())

        word_appearances, reports_lens = _get_stats(dataset)

        self.words_idx, self.word_weights = zip(*word_appearances[:k_first])

        self.lens, self.lens_weights = zip(*reports_lens.items())

    def forward(self, images, reports=None, free=False, **unused_kwargs):
        batch_size = images.size()[0]
        device = images.device

        if reports is None or free:
            n_words = random.choices(self.lens, self.lens_weights, k=1)[0]
        else:
            n_words = reports.size()[-1]

        reports = [
            random.choices(self.words_idx, self.word_weights, k=n_words)
            for _ in range(batch_size)
        ]
        reports = torch.tensor(reports, device=device) # pylint: disable=not-callable
        reports = one_hot(reports, num_classes=self.vocab_size).float()
        return (reports,)
