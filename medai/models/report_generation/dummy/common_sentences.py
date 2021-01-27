import random
from collections import defaultdict, Counter
import torch
from torch import nn
from torch.nn.functional import pad, one_hot
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from medai.utils.nlp import split_sentences_and_pad

def _get_stats(dataset):
    sentence_appearances = defaultdict(lambda: 0) # TODO: use a Counter here?
    sentence_counts = Counter()
    for report in dataset.reports:
        report = split_sentences_and_pad(report['tokens_idxs'])
        sentence_counts[len(report)] += 1
        for sentence in report:
            sentence = sentence.tolist()
            sentence = np.trim_zeros(sentence, 'b')
            sentence = ' '.join(str(val) for val in sentence)

            sentence_appearances[sentence] += 1

    sentence_appearances = sorted(list(sentence_appearances.items()),
                                  key=lambda x: x[1],
                                  reverse=True)

    return sentence_appearances, sentence_counts


class MostCommonSentences(nn.Module):
    def __init__(self, dataset, k_first=100):
        super().__init__()

        self.vocab_size = len(dataset.get_vocab())

        sentence_appearances, sentence_counts = _get_stats(dataset)

        self.sentences, self.weights = zip(*sentence_appearances[:k_first])
        self.n_sentences, self.n_weights = zip(*sentence_counts.items())

    def forward(self, images, reports=None, free=False, **unused_kwargs):
        if reports is None:
            # TODO: handle this case
            raise NotImplementedError

        device = images.device
        batch_size, gt_n_sentences, gt_n_words = reports.size()

        # Generate samples one by one
        reports = []
        for _ in range(batch_size):
            sentences = []
            if free:
                n_sentences = random.choices(self.n_sentences, self.n_weights, k=1)[0]
            else:
                n_sentences = gt_n_sentences

            for sentence in random.choices(self.sentences, self.weights, k=n_sentences):
                sentence = [int(word) for word in sentence.split()]
                sentence = torch.tensor(sentence) # pylint: disable=not-callable
                # shape: n_words

                if not free:
                    if len(sentence) >= gt_n_words:
                        sentence = sentence[:gt_n_words]
                    else:
                        sentence = pad(sentence, (0, gt_n_words - len(sentence)))
                    # shape: gt_n_words

                sentences.append(sentence)

            # sentences shape (list): n_sentences, n_words (gt_ if not free)

            sentences = pad_sequence(sentences, batch_first=True)
            # tensor shape: n_sentences, n_words

            reports.append(sentences)

        # reports shape (list): batch_size, n_sentences, n_words (gt_ if not free)

        if free:
            # n_sentences and n_words in each item may be different
            max_n_sentences = max(r.size()[0] for r in reports)
            max_n_words = max(r.size()[1] for r in reports)
            reports = [
                pad(r, (0, max_n_words - r.size()[1], 0, max_n_sentences - r.size()[0]))
                for r in reports
            ]


        reports = pad_sequence(reports, batch_first=True).to(device)
        # tensor shape: batch_size, n_sentences, n_words

        reports = one_hot(reports, num_classes=self.vocab_size).float()
        # shape: batch_size, n_sentences, n_words, vocab_size

        batch_size, n_sentences = reports.size()[:2]
        dummy_stops = torch.zeros(batch_size, n_sentences, device=device)
        dummy_att_scores = torch.zeros(batch_size, n_sentences, 1, 1, device=device)

        return reports, dummy_stops, dummy_att_scores
