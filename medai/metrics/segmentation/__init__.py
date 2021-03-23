import torch
from ignite.utils import to_onehot
from ignite.metrics import RunningAverage

from medai.metrics.segmentation.iou import IoU
from medai.metrics.segmentation.iobb import IoBB
from medai.metrics.segmentation.dice import Dice
from medai.utils.metrics import attach_metric_for_labels


def _transform_max_onehot(output):
    activations = output['activations'] # shape: BS, n_labels, height, width
    gt_map = output['gt_map'] # shape: BS, height, width
    gt_valid = output.get('gt_valid', None)

    n_labels = activations.size(1)

    _, activations = torch.max(activations, dim=1) # shape: BS, height, width
    activations = to_onehot(activations, n_labels) # shape: BS, n_labels, height, width

    gt_map = to_onehot(gt_map, n_labels) # shape: BS, n_labels, height, width

    return activations, gt_map, gt_valid


def _extract_multilabel(output):
    activations = output['activations']
    gt_map = output['gt_map']
    gt_valid = output.get('gt_valid', None)

    return activations, gt_map, gt_valid


def attach_metrics_segmentation(engine, labels, multilabel=False, device='cuda'):
    loss = RunningAverage(output_transform=lambda x: x['loss'], alpha=1)
    loss.attach(engine, 'loss')

    if multilabel:
        transform = _extract_multilabel
    else:
        transform = _transform_max_onehot

    iou = IoU(reduce_sum=False, output_transform=transform, device=device)
    attach_metric_for_labels(engine, labels, iou, 'iou')

    dice = Dice(output_transform=transform, device=device)
    attach_metric_for_labels(engine, labels, dice, 'dice')
