import torch
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

from medai.utils import divide_tensors

class IoU(Metric):
    """Computes intersection-over-union in images."""
    def __init__(self, reduce_sum=False, **kwargs):
        if reduce_sum:
            self._reduction = torch.sum
        else:
            self._reduction = None

        super().__init__(**kwargs)

    @reinit__is_reduced
    def reset(self):
        self._added_iou = 0
        self._n_samples = 0

    @reinit__is_reduced
    def update(self, output):
        """Updates its internal count.

        Args:
          activations: tensor of shape (batch_size, n_labels, height, width)
          gt_map: tensor of shape (batch_size, n_labels, height, width)
          gt_valid: tensor of shape (batch_size, n_labels)
        """
        activations, gt_map, gt_valid = output

        intersection = (gt_map * activations).sum(dim=(-2, -1))
        # shape: (batch_size, n_labels)

        union = (gt_map + activations).clamp(max=1)
        # shape: (batch_size, n_labels, height, width)
        # assert ((union == 1) | (union == 0)).all(), 'Union wrong: all should be 0 or 1'

        union = union.sum(dim=(-2, -1))
        # shape: (batch_size, n_labels)

        iou = divide_tensors(intersection, union)
        # shape: (batch_size, n_labels)

        if gt_valid is not None:
            iou = iou * gt_valid # Only keep valid bboxes scores
            # shape: (batch_size, n_labels)
            n_samples = gt_valid.sum(dim=0) # shape: n_labels
        else:
            # Assume all are valid
            batch_size, n_labels, _, _ = gt_map.size()
            n_samples = torch.ones(n_labels).to(gt_map.device) * batch_size

        added_iou = iou.sum(dim=0) # shape: n_labels

        if self._reduction:
            added_iou = self._reduction(added_iou) # shape: 1
            n_samples = self._reduction(n_samples) # shape: 1

        self._added_iou += added_iou
        self._n_samples += n_samples


    @sync_all_reduce('_n_samples', '_added_iou')
    def compute(self):
        return divide_tensors(self._added_iou, self._n_samples)