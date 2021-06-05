import logging
import torch
from torch import nn
from torch.nn.functional import interpolate

from medai.losses import get_loss_function

LOGGER = logging.getLogger(__name__)

def get_step_fn_cls_spatial(model, optimizer=None, training=True,
                            cl_lambda=1, spatial_lambda=1, cl_loss_name='bce',
                            out_of_target_only=True, device='cuda'):
    cl_loss_fn = get_loss_function(cl_loss_name)
    spatial_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def step_fn(unused_engine, data_batch):
        # Move inputs to GPU
        images = data_batch.image.to(device)
        # shape: batch_size, channels=3, height, width

        gt_labels = data_batch.labels.to(device)
        # shape(cl_multilabel=True): batch_size, n_labels

        gt_masks = data_batch.masks.to(device).float()
        # shape(seg_multilabel=True): batch_size, n_labels, height, width

        # Enable training
        model.train(training)
        torch.set_grad_enabled(training)

        # zero the parameter gradients
        if training:
            optimizer.zero_grad()

        # Forward
        pred_labels, pred_masks = model(images)
        # pred_labels shape: batch_size, n_labels
        # pred_masks shape: batch_size, n_labels, f-height, f-width

        # Compute classification loss
        gt_labels = gt_labels.float()
        cl_loss = cl_loss_fn(pred_labels, gt_labels) # shape: 1

        # Resize predicted masks
        masks_size = gt_masks.size()[-2:]
        pred_masks = interpolate(pred_masks, masks_size, align_corners=False, mode='bilinear')
        # shape: bs, n_labels, height, width

        # Compute spatial loss
        spatial_loss = spatial_loss_fn(pred_masks, gt_masks)
        # shape: bs, n_labels, height, width

        # Keep only
        if out_of_target_only:
            spatial_loss = spatial_loss[gt_masks == 0]
            # shape: n_values

        if len(spatial_loss) == 0:
            spatial_loss = spatial_loss.sum()
        else:
            spatial_loss = spatial_loss.mean()
        # shape: 1

        # Compute total loss
        total_loss = cl_lambda * cl_loss + spatial_lambda * spatial_loss

        if training:
            total_loss.backward()
            optimizer.step()

        # NOTE: multilabel metrics assume output is sigmoided
        pred_labels = torch.sigmoid(pred_labels.detach())
        pred_masks = torch.sigmoid(pred_masks.detach())

        return {
            'loss': total_loss.detach(),
            'cl_loss': cl_loss.detach(),
            'spatial_loss': spatial_loss.detach(),
            'pred_labels': pred_labels,
            'gt_labels': gt_labels,
            'activations': pred_masks,
            'gt_activations': gt_masks,
        }

    return step_fn
