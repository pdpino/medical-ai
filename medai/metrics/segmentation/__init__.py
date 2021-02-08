import operator
from functools import partial
import torch
from ignite.utils import to_onehot
from ignite.metrics import RunningAverage, MetricsLambda

from medai.metrics.segmentation.iou import IoU
from medai.metrics.segmentation.iobb import IoBB
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

    if average:
        metric_average = MetricsLambda(lambda x: x.mean().item(), metric)
        metric_average.attach(engine, metric_name)


def attach_metrics_segmentation(engine, labels, multilabel=False, device='cuda'):
    loss = RunningAverage(output_transform=lambda x: x['loss'], alpha=1)
    loss.attach(engine, 'loss')

    if multilabel:
        transform = _extract_multilabel
    else:
        transform = _transform_max_onehot

    iou = IoU(reduce_sum=False, output_transform=transform, device=device)
    _attach_metric_for_label(engine, labels, iou, 'iou')

    dice = Dice(output_transform=transform, device=device)
    _attach_metric_for_label(engine, labels, dice, 'dice')


def attach_metrics_image_saliency(engine, labels, keys, multilabel=True):
    if not multilabel:
        raise NotImplementedError()

    def _extract_maps(output, key_gt, key_valid=None):
        activations = output['activations']

        gt_map = output[key_gt]
        gt_valid = output.get(key_valid, None)

        return activations, gt_map, gt_valid

    for (name, key_gt, key_valid) in keys:
        iou = IoU(
            reduce_sum=False,
            output_transform=partial(_extract_maps, key_gt=key_gt, key_valid=key_valid))
        _attach_metric_for_label(engine, labels, iou, f'iou-{name}')

        iobb = IoBB(
            reduce_sum=False,
            output_transform=partial(_extract_maps, key_gt=key_gt, key_valid=key_valid))
        _attach_metric_for_label(engine, labels, iobb, f'iobb-{name}')
