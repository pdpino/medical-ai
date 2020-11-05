import operator
import torch
from ignite.utils import to_onehot
from ignite.metrics import RunningAverage, MetricsLambda

from medai.metrics.segmentation.iou import IoU
from medai.metrics.segmentation.dice import Dice


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


def _attach_metric_for_label(engine, labels, metric, metric_name, average=True):
    for index, label in enumerate(labels):
        metric_for_label_i = MetricsLambda(operator.itemgetter(index), metric)
        metric_for_label_i.attach(engine, f'{metric_name}-{label}')

    metric_average = MetricsLambda(lambda x: x.mean().item(), metric)
    metric_average.attach(engine, metric_name)


def attach_metrics_segmentation(engine, labels, multilabel=False, device='cuda'):
    loss = RunningAverage(output_transform=lambda x: x['loss'], alpha=1)
    loss.attach(engine, 'loss')


    if multilabel:
        transform = _extract_multilabel
    else:
        transform = _transform_max_onehot

    iou = IoU(n_labels=len(labels), output_transform=transform, device=device)
    _attach_metric_for_label(engine, labels, iou, 'iou')

    dice = Dice(n_labels=len(labels), output_transform=transform, device=device)
    _attach_metric_for_label(engine, labels, dice, 'dice')
