import torch
from ignite.metrics import Metric
from torch.nn.functional import mse_loss

from medai.utils import divide_tensors


class HeatmapMSE(Metric):
    """MSE over heatmap images.

    Returns results separated over positive and negative pixels
    """
    def reset(self):
        super().reset()

        self._accum_positives = 0
        self._accum_negatives = 0
        self._accum_total = 0
        self._n_samples = 0


    def _sum_loss_with_filter(self, total_loss, gt_masks, target_value):
        """Sums the loss for a target value.

        The loss is averaged across (target) pixels, and summed across samples (batch_size).

        E.g. if the target_value is 1, it will average the loss across 1-pixels,
        and will sum across batch_size.
        """
        values_filter = gt_masks == target_value
        n_pixels = values_filter.long().sum(dim=(-1, -2)) # bs, n_labels
        filtered_loss = total_loss * values_filter # bs, n_labels, h, w
        filtered_loss = filtered_loss.sum(dim=(-1,-2)) # bs, n_labels
        filtered_loss = divide_tensors(filtered_loss, n_pixels) # bs, n_labels
        filtered_loss = filtered_loss.sum(dim=0) # n_labels

        return filtered_loss


    def update(self, output):
        """Updates its internal count.

        Args:
            output: tuple of (heatmaps, gt_masks, valid), heatmaps and gt_masks
                tensors of shape (batch_size, n_labels, height, width);
                valid is a tensor of shape (batch_size, n_labels)
        """
        heatmaps, gt_masks, valid = output

        total_loss = mse_loss(heatmaps, gt_masks, reduction='none')
        # shape: batch_size, n_labels, h, w

        if valid is not None:
            total_loss = total_loss * valid.unsqueeze(-1).unsqueeze(-1) # Eliminate not valid labels
            # shape: batch_size, n_labels, h, w
            n_samples = valid.sum(dim=0) # shape: n_labels
        else:
            # Assume all are valid
            batch_size, n_labels, _, _ = total_loss.size()
            n_samples = torch.ones(n_labels, device=total_loss.device) * batch_size

        self._n_samples += n_samples

        # Accumulate total-loss
        total_loss_by_pixel = total_loss.mean(dim=(-1,-2)) # shape: bs, n_labels
        self._accum_total += total_loss_by_pixel.sum(dim=0) # shape: n_labels

        # Accumulate positive and negative losses
        self._accum_positives = self._sum_loss_with_filter(total_loss, gt_masks, 1)
        self._accum_negatives = self._sum_loss_with_filter(total_loss, gt_masks, 0)

    def compute(self):
        return {
            'pos': divide_tensors(self._accum_positives, self._n_samples),
            'neg': divide_tensors(self._accum_negatives, self._n_samples),
            'total': divide_tensors(self._accum_total, self._n_samples),
        }
