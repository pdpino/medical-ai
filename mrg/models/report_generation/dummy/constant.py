import torch
from torch import nn
from torch.nn.functional import one_hot

from mrg.utils.nlp import END_IDX, PAD_IDX

class ConstantReport(nn.Module):
    """Returns a constant report."""
    def __init__(self, vocab, report):
        super().__init__()
        
        self.hierarchical = False
        self.report = [vocab[word] for word in report.split()]
        self.vocab_size = len(vocab)
        
        if self.report[-1] != END_IDX:
            self.report.append(END_IDX)

    def forward(self, features, reports=None, free=False):
        batch_size = features.size()[0]
        device = features.device

        if reports is None or free:
            # TODO: implement free
            pass
        else:
            n_words = reports.size()[-1]

        base_report = list(self.report)
        missing = n_words - len(base_report)
        if missing > 0:
            base_report += [PAD_IDX] * missing
        elif missing < 0:
            base_report = base_report[:n_words]
        
        reports = torch.tensor(base_report).to(device).repeat(batch_size, 1)
        # shape: batch_size, n_words

        reports = one_hot(reports, num_classes=self.vocab_size).float()
        # shape: batch_size, n_words, vocab_size
        
        return reports,