import time
import torch
from torch import nn
from torch.nn.functional import interpolate, softmax
from ignite.engine import Engine, Events
from captum.attr import LayerGradCam

from medai.datasets.common.utils import reduce_masks_for_disease
from medai.metrics.segmentation import attach_metrics_image_saliency
from medai.utils.images import bbox_coordinates_to_map
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


def _get_last_layer(compiled_model):
    model_name = compiled_model.metadata['model_kwargs']['model_name']

    model = compiled_model.model
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model = model.module

    if model_name == 'mobilenet-v2':
        return model.features[-1][0] # -1
    if model_name == 'densenet-121-v2':
        return model.features.conv2 # norm5
    if model_name == 'resnet-50-v2':
        return model.features[-1][-1].conv3 # relu

    # DEPRECATED MODELS
    if model_name == 'mobilenet':
        # return model.base_cnn.features[-1][0] # Last conv
        return model.base_cnn.features[-1][-1] # Actual last
    if model_name == 'densenet-121':
        # return model.base_cnn.features.denseblock4.denselayer16.conv2 # Last conv
        return model.base_cnn.features.norm5 # Actual last
    if model_name == 'resnet-50':
        # return model.base_cnn.layer4[-1].conv3 # Last conv
        return model.base_cnn.layer4[-1].relu # Actual last

    raise Exception(f'Last layer not hardcoded for: {model_name}')


def create_grad_cam(compiled_model, device='cuda', multiple_gpu=False):
    wrapped_model = ModelWrapper(compiled_model.model).to(device)
    if multiple_gpu:
        wrapped_model = nn.DataParallel(wrapped_model)

    layer = _get_last_layer(compiled_model)
    grad_cam = LayerGradCam(wrapped_model, layer)

    return grad_cam


def threshold_attributions(attributions, thresh=0.5):
    """Apply a threshold over attributions.

    Args:
        attributions -- tensor of any shape, with values between 0 and 1
        threshold -- float to apply the threshold
    Returns:
        thresholded attributions, same shape as input
    """
    ones = torch.ones(attributions.size()).to(attributions.device)
    zeros = torch.zeros(attributions.size()).to(attributions.device)
    attributions = torch.where(attributions >= thresh, ones, zeros)

    return attributions


def calculate_attributions(grad_cam, images, label_index, relu=True):
    """Calculate grad-cam attributions for a batch of images.

    Args:
        grad_cam -- LayerGradCam object
        images -- tensor of shape (bs, 3, height, width)
        label_index -- int to calculate the grad-cam to
    Returns:
        attributions, tensor of shape (bs, height, width)
    """
    image_size = images.size()[-2:]

    images.requires_grad = True # Needed for Grad-CAM
    # Notice this is not returned to its original value!

    attributions = grad_cam.attribute(images, label_index, relu_attributions=relu).detach()
    # shape: batch_size, 1, layer_h, layer_w

    attributions = interpolate(attributions, image_size)
    # shape: batch_size, 1, h, w

    attributions = tensor_to_range01(attributions)
    # shape: batch_size, 1, h, w

    attributions = attributions.squeeze(1)
    # shape: batch_size, h, w

    return attributions


def get_step_fn(grad_cam, labels,
                enable_bbox=True, enable_masks=True, relu=True,
                thresh=0.5, device='cuda'):
    def step_fn(unused_engine, batch):
        ## Images
        images = batch.image.to(device)
        # shape: batch_size, 3, height, width

        image_size = images.size()[-2:]

        ## Bboxes
        if enable_bbox:
            bboxes_valid = batch.bboxes_valid.to(device) # shape: batch_size, n_labels
            bboxes = batch.bboxes.to(device).long() # shape: batch_size, n_labels, 4
            bboxes_map = bbox_coordinates_to_map(bboxes, bboxes_valid, image_size)
            # shape: batch_size, n_labels, height, width
        else:
            bboxes_map, bboxes_valid = None, None

        ## Organ masks
        if enable_masks:
            batch_masks = batch.masks.to(device) # shape: batch_size, n_organs, height, width
            masks = torch.stack([
                reduce_masks_for_disease(label, batch_masks) # shape: batch_size, height, width
                for label in labels
            ], dim=1)
            # shape: batch_size, n_labels, height, width
        else:
            masks = None

        ## Calculate attributions
        attributions = []
        for index, _ in enumerate(labels):
            attrs = calculate_attributions(grad_cam, images, index, relu=relu)
            attributions.append(attrs)
        attributions = torch.stack(attributions, dim=1)
        # shape: batch_size, n_labels, h, w

        attributions = threshold_attributions(attributions, thresh=thresh)
        # shape: batch_size, n_labels, h, w

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
    grad_cam = create_grad_cam(compiled_model, device=device, multiple_gpu=multiple_gpu)
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
