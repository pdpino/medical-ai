import random
from collections import Counter, defaultdict
import torch
from torch import nn
from torch.nn.functional import one_hot


def _get_stats(dataset):
    word_appearances = defaultdict(lambda: 0)
    reports_lens = []
    for report in dataset.reports:
        report = report['tokens_idxs']
        for word in report:
            word_appearances[word] += 1
        reports_lens.append(len(report))

    word_appearances = sorted(list(word_appearances.items()),
                              key=lambda x: x[1],
                              reverse=True)

    return word_appearances, reports_lens


class MostCommonWords(nn.Module):
    def __init__(self, dataset, k_first=100):
        super().__init__()

        self.hierarchical = False
        self.vocab_size = len(dataset.get_vocab())

        word_appearances, reports_lens = _get_stats(dataset)

        self.words_idx, self.word_weights = zip(*word_appearances[:k_first])

        # TODO: implement free, randomly select a report size
        # self.lens, self.lens_weights = zip(*Counter(reports_lens).items())
        
    def forward(self, features, reports=None, free=False):
        batch_size = features.size()[0]
        device = features.device

        if reports is None or free:
            # TODO: implement free
            pass
        else:
            n_words = reports.size()[-1]
        
        reports = [
            random.choices(self.words_idx, self.word_weights, k=n_words)
            for _ in range(batch_size)
        ]
        reports = torch.tensor(reports).to(device)
        reports = one_hot(reports, num_classes=self.vocab_size).float()
        return reports,