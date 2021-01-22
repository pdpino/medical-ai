import logging
import torch

from medai.losses.out_of_target import OutOfTargetSumLoss
from medai.training.classification.grad_cam import (
    calculate_attributions_for_labels,
    create_grad_cam,
)
from medai.datasets.common.diseases2organs import reduce_masks_for_diseases

LOGGER = logging.getLogger(__name__)

def get_step_fn(model, loss_fn, optimizer=None, training=True,
                multilabel=True, hint=False, diseases=None,
                device='cuda'):
    """Creates a step function for an Engine."""
    if hint:
        hint_loss_fn = OutOfTargetSumLoss()
        grad_cam = create_grad_cam(model, device=device)
        assert diseases is not None, 'If hint=True, diseases cannot be None'

    def step_fn(unused_engine, data_batch):
        # Move inputs to GPU
        images = data_batch.image.to(device)
        # shape: batch_size, channels=3, height, width

        labels = data_batch.labels.to(device)
        # shape(multilabel=True): batch_size, n_labels
        # shape(multilabel=False): batch_size

        # Enable training
        model.train(training)
        torch.set_grad_enabled(training)

        # zero the parameter gradients
        if training:
            optimizer.zero_grad()

        # Forward
        output_tuple = model(images)
        outputs = output_tuple[0]
        # shape: batch_size, n_labels

        if multilabel:
            labels = labels.float()
        else:
            labels = labels.long()

        # Compute classification loss
        cl_loss = loss_fn(outputs, labels)

        if hint:
            with torch.set_grad_enabled(True):
                images.requires_grad = True

                grad_cam_attrs = calculate_attributions_for_labels(
                    grad_cam, images, diseases,
                    relu=True, create_graph=True,
                )
                # shape: (batch_size, n_diseases, height, width)

            images.requires_grad = False

            masks = reduce_masks_for_diseases(
                diseases,
                data_batch.masks.to(device),
            )
            # shape: (batch_size, n_diseases, height, width)

            hint_loss = hint_loss_fn(grad_cam_attrs, masks) # shape: 1
            if hint_loss.isnan().any():
                hint_loss_value = -1
                LOGGER.error(
                    'Hint loss returned nan, training=%s, masks_is_nan=%s, attrs_is_nan=%s',
                    training,
                    masks.isnan().any(),
                    grad_cam_attrs.isnan().any(),
                )

                total_loss = cl_loss
            else:
                hint_loss_value = hint_loss.item()
                total_loss = cl_loss + hint_loss

        else:
            hint_loss_value = -1

            total_loss = cl_loss

        if training:
            total_loss.backward()
            optimizer.step()

        if multilabel:
            # NOTE: multilabel metrics assume output is sigmoided
            outputs = torch.sigmoid(outputs)

        return {
            'loss': total_loss.item(),
            'cl_loss': cl_loss.item(),
            'hint_loss': hint_loss_value,
            'pred_labels': outputs,
            'gt_labels': labels,
        }

    return step_fn
