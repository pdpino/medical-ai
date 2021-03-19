import os

from medai.metrics.detection.metric import MAPCocoMetric
from medai.utils.files import get_results_folder


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
