import random
from collections import defaultdict
import torch
from torch import nn
from torch.nn.functional import pad, one_hot
import numpy as np

from mrg.utils.nlp import END_OF_SENTENCE_IDX, split_sentences_and_pad

def _get_stats(dataset):
    sentence_appearances = defaultdict(lambda: 0)
    for report in dataset.reports:
        report = split_sentences_and_pad(report['tokens_idxs'])
        for sentence in report:
            sentence = sentence.tolist()
            sentence = np.trim_zeros(sentence, 'b')
            sentence = ' '.join(str(val) for val in sentence)
            
            sentence_appearances[sentence] += 1

    sentence_appearances = sorted(list(sentence_appearances.items()),
                                  key=lambda x: x[1],
                                  reverse=True)

    return sentence_appearances


class MostCommonSentences(nn.Module):
    def __init__(self, dataset, k_first=100):
        super().__init__()

        self.hierarchical = True
        self.vocab_size = len(dataset.get_vocab())

        sentence_appearances = _get_stats(dataset)

        self.sentences, self.weights = zip(*sentence_appearances[:k_first])
        
    def forward(self, features, reports=None):
        if reports is None:
            raise NotImplementedError

        device = features.device
        batch_size, n_sentences, n_words = reports.size()
        
        reports = []
        for _ in range(batch_size):
            sentences = []
            for sentence in random.choices(self.sentences, self.weights, k=n_sentences):
                sentence = [int(word) for word in sentence.split()]
                
                if len(sentence) >= n_words:
                    sentence = torch.tensor(sentence[:n_words])
                else:
                    sentence = pad(torch.tensor(sentence), (0, n_words - len(sentence)))
                sentences.append(sentence)
                
            sentences = torch.stack(sentences)
            reports.append(sentences)

        reports = torch.stack(reports).to(device)
        reports = one_hot(reports, num_classes=self.vocab_size).float()
        
        stops = torch.zeros(batch_size, n_sentences).to(device)
        return reports, stops