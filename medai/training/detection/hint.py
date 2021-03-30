import logging
import torch

from medai.training.classification.grad_cam import (
    calculate_attributions_for_labels,
    create_grad_cam,
)

LOGGER = logging.getLogger(__name__)

_CHECK_NAN_OR_INF = False

def get_step_fn_hint(model, cl_loss_fn, hint_loss_fn, h2bb_method=None,
                     optimizer=None, training=True,
                     hint_lambda=1, cl_lambda=1, device='cuda'):
    """Creates a step function for an Engine."""
    multilabel = True
    grad_cam = create_grad_cam(model, device=device)

    if h2bb_method is None:
        LOGGER.warning('h2bb_method is not defined, output will not be complete')

    def step_fn(unused_engine, data_batch):
        # Move inputs to GPU
        images = data_batch.image.to(device)
        # shape: batch_size, channels=3, height, width

        labels = data_batch.labels.to(device)
        # shape(multilabel=True): batch_size, n_labels

        masks = data_batch.masks.to(device)
        # shape: batch_size, n_diseases, height, width


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

        # Compute classification loss
        labels = labels.float()
        cl_loss = cl_loss_fn(outputs, labels)

        # Compute grad-cam
        n_diseases = masks.size(1)
        with torch.set_grad_enabled(True):
            images.requires_grad = True

            grad_cam_attrs = calculate_attributions_for_labels(
                grad_cam, images, range(n_diseases),
                relu=True, create_graph=True,
                resize=True, norm=True,
            )
            # shape: (batch_size, n_diseases, layer-height, layer-width)

        images.requires_grad = False

        # Compute hint_loss
        hint_loss = hint_loss_fn(grad_cam_attrs, masks.float()) # shape: 1

        # Compute total loss
        total_loss = cl_lambda * cl_loss + hint_lambda * hint_loss

        if training:
            total_loss.backward()
            optimizer.step()

        if multilabel:
            # NOTE: multilabel metrics assume output is sigmoided
            outputs = torch.sigmoid(outputs)

        # NOTE: If resize=False, resize manually,
        # as metrics assume heatmaps are the same size as the image
        # image_size = masks.size()[-2:]
        # grad_cam_attrs = resize(grad_cam_attrs, image_size)

        if _CHECK_NAN_OR_INF:
            if total_loss.isnan().any() or total_loss.isinf().any():
                LOGGER.error(
                    'Total loss is nan or inf=%s, cl=%s, hint=%s',
                    total_loss.item(), cl_loss.item(), hint_loss.item(),
                )
                raise Exception('Total loss is nan or inf')

        outputs = outputs.detach()
        grad_cam_attrs = grad_cam_attrs.detach()

        if h2bb_method is not None:
            predictions = h2bb_method(outputs, grad_cam_attrs, data_batch.original_size)
        else:
            predictions = []

        return {
            'loss': total_loss.item(),
            'cl_loss': cl_loss.item(),
            'hint_loss': hint_loss.item(),
            'pred_labels': outputs,
            'gt_labels': labels,
            'activations': grad_cam_attrs,
            'gt_activations': masks,
            'image_fnames': data_batch.image_fname,
            'coco_predictions': predictions,
        }

    return step_fn
