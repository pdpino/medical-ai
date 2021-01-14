import torch
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

from medai.utils import divide_tensors


class IoBB(Metric):
    """Computes intersection-over-bounding-box in images.

    Notice the GT may have any shape, does not need to be a bounding-box.
    """
    def __init__(self, reduce_sum=False, **kwargs):
        if reduce_sum:
            self._reduction = torch.sum
        else:
            self._reduction = None

        super().__init__(**kwargs)

    @reinit__is_reduced
    def reset(self):
        self._added_iobb = 0
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

        bouding_box = gt_map.sum(dim=(-2, -1))
        # shape: (batch_size, n_labels)

        iobb = divide_tensors(intersection, bouding_box)
        # shape: (batch_size, n_labels)

        if gt_valid is not None:
            iobb = iobb * gt_valid # Only keep valid bboxes scores
            # shape: (batch_size, n_labels)
            n_samples = gt_valid.sum(dim=0) # shape: n_labels
        else:
            # Assume all are valid
            batch_size, n_labels, _, _ = gt_map.size()
            n_samples = torch.ones(n_labels).to(gt_map.device) * batch_size

        added_iobb = iobb.sum(dim=0) # shape: n_labels

        if self._reduction:
            added_iobb = self._reduction(added_iobb) # shape: 1
            n_samples = self._reduction(n_samples) # shape: 1

        self._added_iobb += added_iobb
        self._n_samples += n_samples

    @sync_all_reduce('_n_samples', '_added_iobb')
    def compute(self):
        return divide_tensors(self._added_iobb, self._n_samples)