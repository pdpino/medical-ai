import random
import torch
from torch import nn
from torch.nn.functional import pad, one_hot
from torch.nn.utils.rnn import pad_sequence


def _assert_reports_amount(dataset, reports_by_id):
    """Assert that iter_reports_only() method returns the correct amount of reports."""
    n_dataset = len(dataset)
    n_reports_only = len(reports_by_id)
    if n_dataset != n_reports_only:
        raise Exception(
            f'Random received wrong reports amount: expected={n_dataset}, got={n_reports_only}',
        )


class RandomReport(nn.Module):
    """Returns a random report from a dataset."""
    def __init__(self, dataset):
        super().__init__()

        self._prepare_for_random_selection(dataset)

        self.vocab_size = len(dataset.get_vocab())

    def _prepare_for_random_selection(self, dataset):
        self.reports_by_id = dataset.get_reports_by_id()
        assert isinstance(self.reports_by_id, dict)
        _assert_reports_amount(dataset, self.reports_by_id)

        self.report_choices = list(self.reports_by_id.keys())

    def _get_random_reports(self, n_reports, device='cuda'):
        idxs_chosen = random.choices(self.report_choices, k=n_reports)

        reports = [
            # pylint: disable=not-callable
            torch.tensor(self.reports_by_id[idx], device=device)
            for idx in idxs_chosen
        ]
        return reports

    def forward(self, images, reports=None, free=False, **unused_kwargs):
        batch_size = images.size(0)
        device = images.device

        output_reports = self._get_random_reports(batch_size, device=device)
        # list of lists

        output_reports = pad_sequence(output_reports, batch_first=True)
        # tensor of shape batch_size, n_words

        if reports is not None and not free:
            n_words_target = reports.size(1)
            n_words_current = output_reports.size(1)

            if n_words_current >= n_words_target:
                output_reports = output_reports[:, :n_words_target]
            else:
                missing = n_words_target - n_words_current
                output_reports = pad(output_reports, (0, missing))

            # shape: batch_size, n_words_target, vocab_size

        output_reports = one_hot(output_reports, num_classes=self.vocab_size).float()
        # shape: batch_size, n_words, vocab_size

        return (output_reports,)
