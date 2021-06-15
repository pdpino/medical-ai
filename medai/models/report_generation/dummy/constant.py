import torch
from torch import nn
from torch.nn.functional import one_hot

from medai.utils.nlp import END_IDX, PAD_IDX

_IU_DUMMY_REPORT = """the heart is normal in size . the mediastinum is unremarkable .
the lungs are clear .
there is no pneumothorax or pleural effusion . no focal airspace disease .
no pleural effusion or pneumothorax . END"""
_MIMIC_DUMMY_REPORT = 'no acute cardiopulmonary process . END'
_MIMIC_DUMMY_REPORT_2 = """heart size is normal . mediastinum is normal .
lungs are clear . there is no pleural effusion or pneumothorax . no focal consolidation . END"""


_CONSTANT_REPORTS = {
    'iu': _IU_DUMMY_REPORT,
    'mimic': _MIMIC_DUMMY_REPORT,
    'mimic-v2': _MIMIC_DUMMY_REPORT_2,
}

AVAILABLE_CONSTANT_VERSIONS = list(_CONSTANT_REPORTS)

class ConstantReport(nn.Module):
    """Returns a constant report."""
    def __init__(self, vocab, version='iu'):
        super().__init__()

        report = _CONSTANT_REPORTS[version]

        self.report = [vocab[word] for word in report.split()]
        self.vocab_size = len(vocab)

        if self.report[-1] != END_IDX:
            self.report.append(END_IDX)

    def forward(self, images, reports=None, free=False, **unused_kwargs):
        batch_size = images.size()[0]
        device = images.device

        base_report = list(self.report)

        if reports is None or free:
            n_words = len(base_report)
        else:
            n_words = reports.size()[-1]

        missing = n_words - len(base_report)
        if missing > 0:
            base_report += [PAD_IDX] * missing
        elif missing < 0:
            base_report = base_report[:n_words]

        # pylint: disable=not-callable
        reports = torch.tensor(base_report, device=device).repeat(batch_size, 1)
        # shape: batch_size, n_words

        reports = one_hot(reports, num_classes=self.vocab_size).float()
        # shape: batch_size, n_words, vocab_size

        return (reports,)
