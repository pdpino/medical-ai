import os

from medai.metrics.detection.metric import MAPCocoMetric
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
    activations = threshold_attributions(output['activations'], heat_thresh)
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
