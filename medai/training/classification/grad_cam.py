import time
import torch
from torch import nn
from torch.nn.functional import interpolate, softmax
from ignite.engine import Engine, Events
from captum.attr import LayerGradCam

from medai.models.classification import get_last_layer
from medai.datasets.common.diseases2organs import reduce_masks_for_diseases
from medai.metrics.classification import attach_metrics_image_saliency
from medai.utils.images import bbox_coordinates_to_map
from medai.utils.heatmaps import threshold_attributions
from medai.utils import tensor_to_range01


class ModelWrapper(nn.Module):
    """Wraps a model to pass it through captum.LayerGradCam."""
    def __init__(self, model, activation=None):
        super().__init__()
        self.model = model

        self._activation = activation

    def forward(self, x):
        output = self.model(x)[0]
        if self._activation == 'sigmoid':
            output = torch.sigmoid(output)
        elif self._activation == 'softmax':
            output = softmax(output, dim=1)

        return output


def create_grad_cam(model, device='cuda', multiple_gpu=False):
    wrapped_model = ModelWrapper(model).to(device)
    if multiple_gpu:
        raise NotImplementedError('Grad-CAM with multiple_gpu=True is not implemented')
        # wrapped_model = nn.DataParallel(wrapped_model)

    layer = get_last_layer(model)
    grad_cam = LayerGradCam(wrapped_model, layer)

    return grad_cam


def calculate_attributions(grad_cam, images, label_index,
                           relu=True, create_graph=False):
    """Calculate grad-cam attributions for a batch of images.

    Args:
        grad_cam -- LayerGradCam object
        images -- tensor of shape (bs, 3, height, width)
        label_index -- int to calculate the grad-cam to
    Returns:
        attributions, tensor of shape (bs, height, width)
    """
    image_size = images.size()[-2:]

    attributions = grad_cam.attribute(
        images, label_index,
        relu_attributions=relu, create_graph=create_graph,
    )
    # shape: batch_size, 1, layer_h, layer_w

    attributions = interpolate(attributions, image_size)
    # shape: batch_size, 1, h, w

    attributions = tensor_to_range01(attributions)
    # shape: batch_size, 1, h, w

    attributions = attributions.squeeze(1)
    # shape: batch_size, h, w

    return attributions


def calculate_attributions_for_labels(grad_cam, images, labels, **kwargs):
    """Calls calculate_attributions() for multiple labels,
    and stack the results."""
    if isinstance(labels, int):
        labels = range(labels)

    return torch.stack([
        calculate_attributions(grad_cam, images, index, **kwargs) # (bs, h, w)
        for index, _ in enumerate(labels)
    ], dim=1)


def get_step_fn(grad_cam, labels,
                enable_bbox=True, enable_masks=True, relu=True,
                thresh=0.5, device='cuda'):
    """Returns a step_fn that only runs grad_cam evaluation."""
    torch.set_grad_enabled(False)

    def step_fn(unused_engine, batch):
        ## Images
        images = batch.image.to(device)
        # shape: batch_size, 3, height, width

        ## Bboxes
        if enable_bbox:
            image_size = images.size()[-2:]
            bboxes_valid = batch.bboxes_valid.to(device) # shape: batch_size, n_labels
            bboxes_map = bbox_coordinates_to_map(
                batch.bboxes.to(device).long(), # shape: batch_size, n_labels, 4
                bboxes_valid,
                image_size,
            )
            # shape: batch_size, n_labels, height, width
        else:
            bboxes_map, bboxes_valid = None, None

        ## Organ masks
        if enable_masks:
            masks = reduce_masks_for_diseases(
                labels,
                batch.masks.to(device), # shape (bs, n_organs, h, w)
            ) # shape (bs, n_labels, h, w)
        else:
            masks = None

        ## Calculate attributions
        with torch.set_grad_enabled(True):
            images.requires_grad = True # Needed for Grad-CAM
            attributions = calculate_attributions_for_labels(
                grad_cam, images, labels, relu=relu, create_graph=False,
            ).detach()
            # shape: batch_size, n_labels, h, w

            attributions = threshold_attributions(attributions, thresh=thresh)
            # shape: batch_size, n_labels, h, w

            images.requires_grad = False

        return {
            'activations': attributions,
            'bboxes_map': bboxes_map,
            'bboxes_valid': bboxes_valid,
            'masks': masks,
        }

    return step_fn


def create_grad_cam_evaluator(trainer,
                              compiled_model,
                              dataloaders,
                              tb_writer,
                              thresh=0.5,
                              device='cuda',
                              multiple_gpu=False):
    # Get labels
    labels = dataloaders[0].dataset.labels

    # Prepare grad_cam
    grad_cam = create_grad_cam(
        compiled_model.model, device=device, multiple_gpu=multiple_gpu,
    )
    grad_cam_engine = Engine(get_step_fn(grad_cam,
                                         labels,
                                         enable_bbox=False,
                                         thresh=thresh,
                                         device=device))

    keys = [('masks', 'masks', None)]
    attach_metrics_image_saliency(grad_cam_engine, labels, keys, multilabel=True, device=device)


    def run_grad_cam_evaluation(main_engine):
        for dataloader in dataloaders:
            # Run engine
            grad_cam_engine.run(dataloader, 1)

            # Write to TB
            metrics = grad_cam_engine.state.metrics
            dataset_type = dataloader.dataset.dataset_type
            epoch = main_engine.state.epoch
            wall_time = time.time()
            tb_writer.write_metrics(metrics, dataset_type, epoch, wall_time)


    trainer.add_event_handler(Events.EPOCH_COMPLETED, run_grad_cam_evaluation)


def calculate_cam(model, x):
    """Calculates CAM manually with a CNN over images.

    Args:
        model -- CNN with `features` (any convolutional configuration)
            and `prediction`, a single Linear layer.
        x -- input images, tensor of shape: batch_size, 3, height, width
    Returns:
        CAM activations, tensor of shape
            (batch_size, n_diseases, features-height, features-width)
    """
    x = model.features(x)
    # shape: batch_size, n_features, features-h, features-w

    weights, unused_bias = list(model.prediction.parameters())
    # weights shape: n_diseases, n_features

    activations = torch.matmul(weights, x.transpose(1, 2)).transpose(1, 2)
    # shape: batch_size, n_diseases, features-h, features-w

    return activations
