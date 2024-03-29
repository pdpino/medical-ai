import json
import os
import operator
import logging
import torch
import pandas as pd
from ignite.metrics import RunningAverage, MetricsLambda

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


def _get_results_fpath(run_id, suffix='', **kwargs):
    folder = get_results_folder(run_id, **kwargs)

    filename = 'metrics'
    if suffix:
        filename += f'-{suffix}'

    filepath = os.path.join(folder, f'{filename}.json')

    return filepath


# TODO: rename as "are_metrics_saved", is more accurate
def are_results_saved(run_id, suffix=''):
    filepath = _get_results_fpath(run_id, suffix=suffix, save_mode=False)

    return filepath is not None and os.path.isfile(filepath)


def save_results(metrics_dict, run_id,
                 suffix='', merge_prev=True):
    filepath = _get_results_fpath(run_id, suffix=suffix, save_mode=True)

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
        MetricsLambda(lambda x: x.item(), loss_metric).attach(engine, loss_name)
