import json
import os
import torch

from medai.utils.common import WORKSPACE_DIR

class MetricsEncoder(json.JSONEncoder):
    """Serializes metrics.

    ConfusionMatrix metric returns a torch.Tensor, which is not serializable
        --> transform to list of lists.
    """
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().numpy().tolist()
        return obj


def get_results_folder(run_name, classification=True, debug=True, save_mode=False):
    mode_folder = 'classification' if classification else 'report_generation'
    debug_folder = 'debug' if debug else ''

    folder = os.path.join(WORKSPACE_DIR, mode_folder, 'results', debug_folder)
    folder = os.path.join(folder, run_name)

    if save_mode:
        os.makedirs(folder, exist_ok=True)
    else:
        assert os.path.isdir(folder), f'Run folder does not exist: {folder}'

    return folder


def save_results(metrics_dict, run_name, classification=True, debug=True,
                 suffix='', merge_prev=True):
    folder = get_results_folder(run_name, classification=classification, debug=debug,
                                 save_mode=True)

    filename = 'metrics'
    if suffix:
        filename += f'-{suffix}'

    filepath = os.path.join(folder, f'{filename}.json')

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
    results_folder = get_results_folder(run_name, debug=debug, classification=False)
    suffix = get_free_suffix(free)

    outputs_path = os.path.join(results_folder, f'outputs-{suffix}.csv')

    if not os.path.isfile(outputs_path):
        print('Outputs file not found: ', outputs_path)
        return None

    return pd.read_csv(
        outputs_path,
        keep_default_na=False, # Do not treat the empty-string as NaN value
    )
