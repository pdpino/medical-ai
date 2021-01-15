import time
import torch
from torch import nn
from torch.nn.functional import interpolate
from ignite.engine import Engine, Events
from captum.attr import LayerGradCam

from medai.datasets.common.utils import reduce_masks_for_disease
from medai.metrics.segmentation import attach_metrics_image_saliency
from medai.utils.images import bbox_coordinates_to_map
from medai.utils import tensor_to_range01


class ModelWrapper(nn.Module):
    """Wraps a model to pass it through captum.LayerGradCam."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)[0]
        output = torch.sigmoid(output)
        return output


def _get_last_layer(compiled_model):
    model_name = compiled_model.metadata['model_kwargs']['model_name']

    model = compiled_model.model
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model = model.module

    if model_name == 'mobilenet-v2':
        return model.features[-1][-1]
    if model_name == 'densenet-121-v2':
        return model.features.norm5
    if model_name == 'resnet-50-v2':
        return model.features.layer4[-1].relu

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
    ones = torch.ones(attributions.size()).to(attributions.device)
    zeros = torch.zeros(attributions.size()).to(attributions.device)
    attributions = torch.where(attributions >= thresh, ones, zeros)
    # shape: batch_size, h, w

    return attributions


def calculate_attributions(grad_cam, images, label_index, image_size):
    attributions = grad_cam.attribute(images, label_index).detach()
    # shape: batch_size, 1, layer_h, layer_w

    attributions = interpolate(attributions, image_size)
    # shape: batch_size, 1, h, w

    attributions = tensor_to_range01(attributions)
    # shape: batch_size, 1, h, w

    attributions = attributions.squeeze(1)
    # shape: batch_size, h, w

    return attributions


def get_step_fn(grad_cam, labels,
                enable_bbox=True, enable_masks=True,
                thresh=0.5, device='cuda'):
    def step_fn(unused_engine, batch):
        ## Images
        images = batch.image.to(device)
        images.requires_grad = True # Needed for Grad-CAM
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
            attrs = calculate_attributions(grad_cam, images, index, image_size)
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
