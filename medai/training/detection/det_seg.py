import logging
import torch

LOGGER = logging.getLogger(__name__)

def get_step_fn_det_seg(model, cl_loss_fn, seg_loss_fn, h2bb_method,
                        optimizer=None, training=True,
                        cl_lambda=1, seg_lambda=1,
                        device='cuda'):
    """Creates a step function for an Engine."""
    def step_fn(unused_engine, data_batch):
        # Move inputs to GPU
        images = data_batch.image.to(device)
        # shape: batch_size, channels=3, height, width

        gt_labels = data_batch.labels.to(device)
        # shape(multilabel=True): batch_size, n_labels

        gt_masks = data_batch.masks.to(device)
        # shape: batch_size, n_diseases, height, width

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
        gt_masks = gt_masks.float()
        seg_loss = seg_loss_fn(pred_masks, gt_masks)

        # Compute total loss
        total_loss = cl_lambda * cl_loss + seg_lambda * seg_loss

        if training:
            total_loss.backward()
            optimizer.step()

        pred_labels = pred_labels.detach()
        pred_masks = pred_masks.detach()

        # NOTE: multilabel metrics assume output is sigmoided
        pred_labels = torch.sigmoid(pred_labels)
        pred_masks = torch.sigmoid(pred_masks)

        coco_predictions = h2bb_method(pred_labels, pred_masks)

        return {
            'loss': total_loss.item(),
            'cl_loss': cl_loss.item(),
            'seg_loss': seg_loss.item(),
            'pred_labels': pred_labels,
            'gt_labels': gt_labels,
            'activations': pred_masks,
            'gt_activations': gt_masks,
            'image_fnames': data_batch.image_fname,
            'coco_predictions': coco_predictions,
        }

    return step_fn
