import torch
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

from mrg.utils import PAD_IDX

class WordAccuracy(Metric):
    """Computes accuracy over the words predicted, ignoring padding."""
    def __init__(self, pad_idx=PAD_IDX, output_transform=lambda x: x, device=None):
        self.pad_idx = pad_idx

        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._n_correct = 0
        self._n_samples = 0

        super().reset()

    @reinit__is_reduced
    def update(self, output):
        generated_words, seq = output
        # shape (both arrays): batch_size, sentence_len

        ignore_padding = (seq != self.pad_idx)

        self._n_samples += torch.sum(ignore_padding).item()
        self._n_correct += torch.sum((seq == generated_words) & (ignore_padding)).item()

    @sync_all_reduce('_n_correct', '_n_samples')
    def compute(self):
        return self._n_correct / self._n_samples if self._n_samples > 0 else 0