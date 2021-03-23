import os
from functools import partial

from ignite.metrics import MetricsLambda

from medai.metrics.detection.coco_map.metric import MAPCocoMetric
from medai.metrics.detection.mse import HeatmapMSE
from medai.metrics.segmentation.iou import IoU
from medai.metrics.segmentation.iobb import IoBB
from medai.utils.files import get_results_folder
from medai.utils.heatmaps import threshold_attributions
from medai.utils.metrics import attach_metric_for_labels


def _extract_for_mAP(output):
    image_names = output['image_fnames']
    labels_pred = output['pred_labels']
    heatmaps = output['activations']

    return image_names, labels_pred, heatmaps


def _get_temp_outputs_fpath(run_name, dataset_type, debug=True):
    folder = get_results_folder(run_name,
                                task='det',
                                debug=debug,
                                save_mode=True)
    path = os.path.join(folder, f'temp-outputs-{dataset_type}.csv')

    return path


def attach_mAP_coco(engine, dataloader, run_name, debug=True, device='cuda'):
    gt_df = dataloader.dataset.coco_gt_df
    dataset_type = dataloader.dataset.dataset_type
    temp_fpath = _get_temp_outputs_fpath(run_name, dataset_type, debug=debug)

    metric = MAPCocoMetric(gt_df, temp_fpath,
                           output_transform=_extract_for_mAP, device=device)
    metric.attach(engine, 'mAP')

    return


def _threshold_activations_and_keep_valid(output, cls_thresh=0.3, heat_thresh=0.5):
    """Extracts values for IoX metrics.

    Applies a threshold to activations, and creates a gt_valid array to use only
    TP values for the metric calculation.
    """
    activations = output['activations']
    if heat_thresh is not None:
        activations = threshold_attributions(activations, heat_thresh)

    gt_map = output['gt_activations']

    # Only use TP samples to calculate metrics
    gt_valid = output['gt_labels'].bool() & (output['pred_labels'] > cls_thresh)

    return activations, gt_map, gt_valid


def attach_metrics_iox(engine, labels, multilabel=False, device='cuda'):
    """Attaches IoU, IoBB metrics.

    Expects not-thresholded values.
    """
    if not multilabel:
        raise NotImplementedError()

    iou = IoU(reduce_sum=False,
              output_transform=_threshold_activations_and_keep_valid,
              device=device)
    attach_metric_for_labels(engine, labels, iou, 'iou')

    iobb = IoBB(reduce_sum=False,
                output_transform=_threshold_activations_and_keep_valid,
                device=device)
    attach_metric_for_labels(engine, labels, iobb, 'iobb')


def attach_mse(engine, labels, multilabel=True, device='cuda'):
    if not multilabel:
        raise NotImplementedError()

    mse = HeatmapMSE(
        output_transform=partial(_threshold_activations_and_keep_valid, heat_thresh=None),
        device='cuda',
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
