import json
import os
import operator
import logging
import torch
import pandas as pd
from ignite.metrics import RunningAverage

from medai.utils.files import get_results_folder

LOGGER = logging.getLogger(__name__)

class MetricsEncoder(json.JSONEncoder):
    """Serializes metrics.

    ConfusionMatrix metric returns a torch.Tensor, which is not serializable
        --> transform to list of lists.
    """
    def default(self, obj): # pylint: disable=arguments-differ
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().tolist()
        return obj


def _get_results_fpath(run_name, task, debug=True, suffix='', **kwargs):
    folder = get_results_folder(run_name, task=task, debug=debug, **kwargs)

    filename = 'metrics'
    if suffix:
        filename += f'-{suffix}'

    filepath = os.path.join(folder, f'{filename}.json')

    return filepath


def are_results_saved(run_name, task, debug=True, suffix=''):
    # FIXME: run_name cannot be passed as timestamp-only (throws error)
    filepath = _get_results_fpath(run_name, task, debug=debug, suffix=suffix,
                                  save_mode=False)

    return filepath is not None and os.path.isfile(filepath)


def save_results(metrics_dict, run_name, task, debug=True,
                 suffix='', merge_prev=True):
    filepath = _get_results_fpath(run_name, task, debug=debug, suffix=suffix, save_mode=True)

    if os.path.isfile(filepath) and merge_prev:
        with open(filepath, 'r') as f:
            old_dict = json.load(f)
        metrics_dict = {
            **old_dict,
            **metrics_dict
        }

    with open(filepath, 'w') as f:
        json.dump(metrics_dict, f, cls=MetricsEncoder)

    LOGGER.info('Saved metrics to %s', filepath)


def load_rg_outputs(run_name, debug=True, free=False):
    """Load report-generation output dataframe.

    Returns a DataFrame with columns:
    filename,epoch,dataset_type,ground_truth,generated
    """
    results_folder = get_results_folder(run_name, task='rg', debug=debug)
    suffix = 'free' if free else 'notfree'

    outputs_path = os.path.join(results_folder, f'outputs-{suffix}.csv')

    if not os.path.isfile(outputs_path):
        LOGGER.error('Outputs file not found: %s', outputs_path)
        return None

    return pd.read_csv(
        outputs_path,
        keep_default_na=False, # Do not treat the empty-string as NaN value
    )


def attach_losses(engine, losses=[], device='cuda'):
    """Attaches losses to an engine.

    Includes 'loss' by default.
    """
    losses = ['loss'] + losses

    for loss_name in losses:
        loss_metric = RunningAverage(
            output_transform=operator.itemgetter(loss_name), alpha=1,
            device=device,
        )
        loss_metric.attach(engine, loss_name)
