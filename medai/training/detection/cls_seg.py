import logging
import torch
from torch import nn

from medai.losses import get_loss_function

LOGGER = logging.getLogger(__name__)

def get_step_fn_cls_seg(model, optimizer=None, training=True,
                        cl_lambda=1, seg_lambda=1,
                        cl_loss_name='bce', seg_weights=None,
                        device='cuda'):
    cl_loss_fn = get_loss_function(cl_loss_name)

    if isinstance(seg_weights, (list, tuple)):
        seg_weights = torch.tensor(seg_weights, device=device) # pylint: disable=not-callable
    elif isinstance(seg_weights, torch.Tensor):
        seg_weights = seg_weights.to(device)
    seg_loss_fn = nn.CrossEntropyLoss(weight=seg_weights)

    def step_fn(unused_engine, data_batch):
        # Move inputs to GPU
        images = data_batch.image.to(device)
        # shape: batch_size, channels=3, height, width

        gt_labels = data_batch.labels.to(device)
        # shape(cl_multilabel=True): batch_size, n_labels

        gt_masks = data_batch.masks.to(device)
        # shape(seg_multilabel=False): batch_size, height, width

        # Enable training
        model.train(training)
        torch.set_grad_enabled(training)

        # zero the parameter gradients
        if training:
            optimizer.zero_grad()

        # Forward
        pred_labels, pred_masks = model(images)
        # pred_labels shape: batch_size, n_labels
        # pred_masks shape: batch_size, n_labels, height, width

        # Compute classification loss
        gt_labels = gt_labels.float()
        cl_loss = cl_loss_fn(pred_labels, gt_labels)

        # Compute segmentation loss
        seg_loss = seg_loss_fn(pred_masks, gt_masks)

        # Compute total loss
        total_loss = cl_lambda * cl_loss + seg_lambda * seg_loss

        if training:
            total_loss.backward()
            optimizer.step()

        # NOTE: multilabel metrics assume output is sigmoided
        pred_labels = pred_labels.detach()
        pred_labels = torch.sigmoid(pred_labels)

        return {
            'loss': total_loss.item(),
            'cl_loss': cl_loss.item(),
            'seg_loss': seg_loss.item(),
            'pred_labels': pred_labels,
            'gt_labels': gt_labels,
            'activations': pred_masks.detach(),
            'gt_activations': gt_masks,
        }

    return step_fn
