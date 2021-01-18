import json
import os
import torch
import pandas as pd

from medai.utils.files import get_results_folder

class MetricsEncoder(json.JSONEncoder):
    """Serializes metrics.

    ConfusionMatrix metric returns a torch.Tensor, which is not serializable
        --> transform to list of lists.
    """
    def default(self, obj): # pylint: disable=arguments-differ
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().numpy().tolist()
        return obj


def get_results_fpath(run_name, task, debug=True, suffix='', save_mode=False):
    folder = get_results_folder(run_name, task=task, debug=debug, save_mode=save_mode)

    filename = 'metrics'
    if suffix:
        filename += f'-{suffix}'

    filepath = os.path.join(folder, f'{filename}.json')

    return filepath


def are_results_saved(run_name, task, debug=True, suffix=''):
    filepath = get_results_fpath(run_name, task, debug=debug, suffix=suffix, save_mode=False)

    return os.path.isfile(filepath)


def save_results(metrics_dict, run_name, task, debug=True,
                 suffix='', merge_prev=True):
    filepath = get_results_fpath(run_name, task, debug=debug, suffix=suffix, save_mode=True)

    if os.path.isfile(filepath) and merge_prev:
        with open(filepath, 'r') as f:
            old_dict = json.load(f)
        metrics_dict = {
            **old_dict,
            **metrics_dict
        }

    with open(filepath, 'w') as f:
        json.dump(metrics_dict, f, cls=MetricsEncoder)

    print(f'Saved metrics to {filepath}')


def load_rg_outputs(run_name, debug=True, free=False):
    """Load report-generation output dataframe.

    Returns a DataFrame with columns:
    filename,epoch,dataset_type,ground_truth,generated
    """
    results_folder = get_results_folder(run_name, task='rg', debug=debug)
    suffix = 'free' if free else 'notfree'

    outputs_path = os.path.join(results_folder, f'outputs-{suffix}.csv')

    if not os.path.isfile(outputs_path):
        print('Outputs file not found: ', outputs_path)
        return None

    return pd.read_csv(
        outputs_path,
        keep_default_na=False, # Do not treat the empty-string as NaN value
    )
