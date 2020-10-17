import torch
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class IoU(Metric):
    """Computes intersection-over-union in images."""
    def __init__(self, n_labels=14, output_transform=lambda x: x, device='cuda'):
        self.n_labels = n_labels
        self._device = device # REVIEW: isn't there an internal device attribute?

        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._added_iou = torch.zeros(self.n_labels).to(self._device)
        self._n_samples = torch.zeros(self.n_labels).to(self._device)

    @reinit__is_reduced
    def update(self, output):
        """Updates its internal count.
        
        Args:
          activations: tensor of shape (batch_size, n_labels, height, width)
          gt_map: tensor of shape (batch_size, n_labels, height, width)
          gt_valid: tensor of shape (batch_size, n_labels)
        """
        activations, gt_map, gt_valid = output
        
        intersection = (gt_map * activations).sum(dim=-1).sum(dim=-1)
        # shape: (batch_size, n_labels)
        
        union = (gt_map + activations).clamp(max=1)
        # shape: (batch_size, n_labels, height, width)
        # assert ((union == 1) | (union == 0)).all(), 'Union wrong: all should be 0 or 1'
        
        union = union.sum(dim=-1).sum(dim=-1)
        # shape: (batch_size, n_labels)
        
        iou = intersection / union
        # shape: (batch_size, n_labels)
        
        iou = iou * gt_valid # Only keep valid bboxes scores
        # shape: (batch_size, n_labels)
        
        added_iou = iou.sum(dim=0) # shape: n_labels
        n_samples = gt_valid.sum(dim=0) # shape: n_labels
        
        self._added_iou += added_iou
        self._n_samples += n_samples


    @sync_all_reduce('_n_samples', '_added_iou')
    def compute(self):
        dont_use = self._n_samples == 0
        
        added_iou = self._added_iou.clone()
        added_iou[dont_use] = 0
        
        n_samples = self._n_samples.clone()
        n_samples[dont_use] = 1
        
        return added_iou / n_samples