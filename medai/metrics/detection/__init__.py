import os
from functools import partial
import logging

import torch
from ignite.utils import to_onehot
from ignite.metrics import MetricsLambda

from medai.metrics.detection.coco_writer.writer import get_outputs_fpath, CocoResultsWriter
from medai.metrics.detection.coco_map.metric import MAPCocoMetric
from medai.metrics.detection.mse import HeatmapMSE
from medai.metrics.segmentation.iou import IoU
from medai.metrics.segmentation.iobb import IoBB
from medai.metrics.segmentation.ioo import IoO
from medai.utils.files import get_results_folder
from medai.utils.heatmaps import threshold_attributions
from medai.utils.metrics import attach_metric_for_labels

LOGGER = logging.getLogger(__name__)

def _extract_for_mAP(output, key='coco_predictions'):
    image_names = output['image_fnames']
    predictions = output[key]

    return image_names, predictions


def attach_mAP_coco(engine, dataloader, run_id, suffix=None, device='cuda'):
    if not hasattr(dataloader.dataset, 'coco_gt_df'):
        class_name = dataloader.dataset.__class__.__name__
        LOGGER.warning(
            'Dataset %s does not have a COCO GT dataframe, could not attach mAP metric',
            class_name,
        )
        return
    gt_df = dataloader.dataset.coco_gt_df
    dataset_type = dataloader.dataset.dataset_type

    temp_fpath = get_outputs_fpath(run_id, dataset_type, suffix=suffix)
    writer = CocoResultsWriter(temp_fpath)

    coco_key = 'coco_predictions'
    metric_name = 'mAP'
    if suffix:
        coco_key += f'_{suffix}'
        metric_name += f'-{suffix}'

    metric = MAPCocoMetric(gt_df, writer,
                           donotcompute=(dataset_type == 'test'),
                           output_transform=partial(_extract_for_mAP, key=coco_key),
                           device=device)
    metric.attach(engine, metric_name)


def _threshold_activations_and_keep_valid(output,
                                          cls_thresh=0.3,
                                          heat_thresh=0.5,
                                          only='TP',
                                          ):
    """Extracts values for IoX metrics.

    Applies a threshold to activations, and creates a gt_valid array to use only
    TP/T values for the metric calculation.
    """
    activations = output['activations']
    if heat_thresh is not None:
        activations = threshold_attributions(activations, heat_thresh)

    gt_map = output['gt_activations']

    if only == 'TP':
        # Use only TP samples to calculate metrics
        gt_valid = output['gt_labels'].bool() & (output['pred_labels'] > cls_thresh)
    elif only == 'T':
        # Use only true samples to calculate metric
        gt_valid = output['gt_labels'].bool()
    else:
        gt_valid = None

    return activations, gt_map, gt_valid


def _transform_max_onehot(output):
    """Extracts values for IoX metrics.

    Transforms with onehot encoding (i.e. useful for multilabel=False cases)
    """
    activations = output['activations'] # shape: bs, n_labels, height, width
    gt_map = output['gt_activations'] # shape: bs, height, width

    n_labels = activations.size(1)

    # One-hot activations
    _, activations = torch.max(activations, dim=1) # shape: bs, height, width
    activations = to_onehot(activations, n_labels) # shape: bs, n_labels, height, width

    # One-hot ground-truth
    gt_map = to_onehot(gt_map, n_labels) # shape: bs, n_labels, height, width

    gt_valid = None

    return activations, gt_map, gt_valid


def attach_metrics_iox(engine, labels, multilabel=False,
                       iou=True, iobb=True, ioo=False,
                       device='cuda', **kwargs):
    """Attaches IoU, IoBB, IoO metrics.

    Expects not-thresholded values.
    """
    if multilabel:
        transform_fn = partial(_threshold_activations_and_keep_valid, **kwargs)
    else:
        transform_fn = _transform_max_onehot
        if len(kwargs) > 0:
            LOGGER.warning('Passed kwargs not used: %s', kwargs)

    if iou:
        iou_metric = IoU(
            reduce_sum=False, output_transform=transform_fn, device=device,
        )
        attach_metric_for_labels(engine, labels, iou_metric, 'iou')

    if iobb:
        iobb_metric = IoBB(
            reduce_sum=False, output_transform=transform_fn, device=device,
        )
        attach_metric_for_labels(engine, labels, iobb_metric, 'iobb')

    if ioo:
        ioo_metric = IoO(
            reduce_sum=False, output_transform=transform_fn, device=device,
        )
        attach_metric_for_labels(engine, labels, ioo_metric, 'ioo')


def attach_mse(engine, labels, multilabel=True, device='cuda'):
    if not multilabel:
        raise NotImplementedError()

    mse = HeatmapMSE(
        output_transform=partial(_threshold_activations_and_keep_valid, heat_thresh=None),
        device=device,
    )

    def _i_getter(result, key, index):
        return result[key][index].item()

    def _avg_getter(result, key):
        return result[key].mean().item()

    metric_name = 'mse'
    for key in ['pos', 'neg', 'total']:
        for index, label in enumerate(labels):
            metric_for_label_i = MetricsLambda(partial(_i_getter, key=key, index=index), mse)
            metric_for_label_i.attach(engine, f'{metric_name}-{key}-{label}')

        metric_average = MetricsLambda(partial(_avg_getter, key=key), mse)
        metric_average.attach(engine, f'{metric_name}-{key}')
