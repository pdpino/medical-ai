import logging
import torch

from medai.losses.out_of_target import OutOfTargetSumLoss
from medai.training.classification.grad_cam import (
    calculate_attributions_for_labels,
    create_grad_cam,
)
from medai.datasets.common.diseases2organs import reduce_masks_for_diseases

LOGGER = logging.getLogger(__name__)

_CHECK_NAN_OR_INF = False

def get_step_fn(model, loss_fn, optimizer=None, training=True,
                multilabel=True, hint=False, hint_lambda=1,
                diseases=None,
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

        if _CHECK_NAN_OR_INF:
            is_nan = outputs.isnan().any().item()
            is_inf = outputs.isinf().any().item()
            if is_nan or is_inf:
                LOGGER.error(
                    'Outputs are nan or inf: nan=%s, inf=%s',
                    is_nan, is_inf,
                )
                raise Exception('Outputs reached nan or inf')

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

            masks = reduce_masks_for_diseases( # TODO: move this to collate_fn()
                diseases,
                data_batch.masks.to(device),
            )
            # shape: (batch_size, n_diseases, height, width)

            hint_loss = hint_loss_fn(grad_cam_attrs, masks) # shape: 1
            total_loss = cl_loss + hint_lambda * hint_loss
        else:
            hint_loss = torch.tensor(-1) # pylint: disable=not-callable
            total_loss = cl_loss

            grad_cam_attrs = torch.tensor(-1) # pylint: disable=not-callable
            masks = torch.tensor(-1) # pylint: disable=not-callable


        if training:
            total_loss.backward()
            optimizer.step()

        if multilabel:
            # NOTE: multilabel metrics assume output is sigmoided
            outputs = torch.sigmoid(outputs)

        if _CHECK_NAN_OR_INF:
            if total_loss.isnan().any() or total_loss.isinf().any():
                LOGGER.error(
                    'Total loss is nan or inf=%s, cl=%s, hint=%s',
                    total_loss.item(), cl_loss.item(), hint_loss.item(),
                )
                raise Exception('Total loss is nan or inf')

        return {
            'loss': total_loss.item(),
            'cl_loss': cl_loss.item(),
            'hint_loss': hint_loss.item(),
            'pred_labels': outputs.detach(),
            'gt_labels': labels,
            'activations': grad_cam_attrs.detach(),
            'gt_activations': masks,
        }

    return step_fn
