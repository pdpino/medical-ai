import torch
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

from medai.utils import divide_tensors

class Dice(Metric):
    """Computes Dice coefficient in images."""
    @reinit__is_reduced
    def reset(self):
        self._added_dice = 0
        self._n_samples = 0

    @reinit__is_reduced
    def update(self, output):
        """Updates its internal count.

        Args:
          activations: tensor of shape (batch_size, n_labels, height, width)
          gt_map: tensor of shape (batch_size, n_labels, height, width)
          gt_valid: tensor of shape (batch_size, n_labels) # not used for now
        """
        activations, gt_map, _ = output

        intersection = (2 * gt_map * activations).sum(dim=[2, 3]) # shape: BS, n_labels
        areas = gt_map.sum(dim=[2, 3]) + activations.sum(dim=[2, 3]) # shape: BS, n_labels
        dice = divide_tensors(intersection, areas) # shape: BS, n_labels

        added_dice = dice.sum(dim=0) # shape: n_labels

        batch_size, n_labels, _, _ = gt_map.size()
        n_samples = torch.ones(n_labels).to(self._device) * batch_size # shape: n_labels

        self._added_dice += added_dice
        self._n_samples += n_samples


    @sync_all_reduce('_n_samples', '_added_dice')
    def compute(self):
        return divide_tensors(self._added_dice, self._n_samples)