import random
import torch
from torch import nn
from torch.nn.functional import pad, one_hot
from torch.nn.utils.rnn import pad_sequence

def _extract_reports(dataset):
    reports = []
    for report in dataset.reports:
        report = report['tokens_idxs']
        report = torch.tensor(report)
        reports.append(report)

    return reports

class RandomReport(nn.Module):
    """Returns a random report from a dataset."""
    def __init__(self, dataset):
        super().__init__()
        
        self.reports = _extract_reports(dataset)
        self.vocab_size = len(dataset.get_vocab())

    def forward(self, images, reports=None, free=False, **unused_kwargs):
        batch_size = images.size()[0]
        device = images.device

        output_reports = random.choices(self.reports, k=batch_size)
        # list of lists

        output_reports = pad_sequence(output_reports, batch_first=True)
        # tensor of shape batch_size, n_words

        if reports is not None and not free:
            n_words_target = reports.size()[1]
            n_words_current = output_reports.size()[1]

            if n_words_current >= n_words_target:
                output_reports = output_reports[:, :n_words_target]
            else:
                missing = n_words_target - n_words_current
                output_reports = pad(output_reports, (0, missing))

            # shape: batch_size, n_words_target, vocab_size

        output_reports = one_hot(output_reports, num_classes=self.vocab_size).float()
        # shape: batch_size, n_words, vocab_size

        output_reports = output_reports.to(device)

        return output_reports,