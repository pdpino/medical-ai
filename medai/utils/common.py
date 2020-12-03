from datetime import datetime
import time
import os
import logging
import numpy as np

WORKSPACE_DIR = os.environ['MED_AI_WORKSPACE_DIR']
TMP_DIR = os.path.join(WORKSPACE_DIR, 'tmp')

def get_timestamp(short=True):
    now = datetime.fromtimestamp(time.time())

    if short:
        return now.strftime('%m%d_%H%M%S')
    return now.strftime('%Y-%m-%d_%H-%M-%S')


def duration_to_str(all_seconds):
    all_seconds = int(all_seconds)
    seconds = all_seconds % 60
    minutes = all_seconds // 60
    hours = minutes // 60

    minutes = minutes % 60

    return '{}h {}m {}s'.format(hours, minutes, int(seconds))


def labels_to_str(values, labels, thresh=0.5, sort_values=False, show_value=False):
    positive = [(val, label) for val, label in zip(values, labels) if val > thresh]
    if sort_values:
        positive = sorted(positive, reverse=True)

    if len(positive) > 0:
        format_str = '{1} ({0:.3f})' if show_value else '{1}'
        return ', '.join(format_str.format(val, label) for val, label in positive)
    else:
        return 'No Findings'


def arr_to_range(arr, min_value=0, max_value=1):
    return np.interp(arr, (arr.min(), arr.max()), (min_value, max_value))


def tensor_to_range01(arr, eps=1e-8):
    # arr shape: batch_size, n_channels, height, width

    arr_min = arr.min(-1)[0].min(-1)[0]
    # shape: batch_size, n_channels

    arr_max = arr.max(-1)[0].max(-1)[0]
    # shape: batch_size, n_channels

    arr_min = arr_min.unsqueeze(-1).unsqueeze(-1)
    arr_max = arr_max.unsqueeze(-1).unsqueeze(-1)
    # shape: batch_size, n_channels, 1, 1

    arr_range = (arr_max - arr_min) + eps
    return (arr - arr_min) / arr_range


def divide_tensors(a, b):
    """Divide two tensors element-wise, avoiding NaN values in the result."""
    dont_use = b == 0

    a = a.clone()
    a[dont_use] = 0

    b = b.clone()
    b[dont_use] = 1

    return a.true_divide(b)


def num_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def write_list_to_txt(arr, filepath, sep='\n'):
    """Writes a list of strings to a file"""
    with open(filepath, 'w') as f:
        for line in arr:
            f.write(line + sep)


def parse_str_or_int(s):
    """Parses as int if possible, else str."""
    try:
        return int(s)
    except:
        return s


def print_hw_options(device, args):
    _CUDA_VISIBLE = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    d = {
        'device': device,
        'visible': _CUDA_VISIBLE,
        'multiple': args.multiple_gpu,
        'num_workers': args.num_workers,
        'num_threads': args.num_threads,
    }
    info_str = ' '.join(f'{k}={v}' for k, v in d.items())
    print(f'Using {info_str}')

def config_logging():
    logging.basicConfig(
        level=logging.WARNING,
        format='%(levelname)s(%(asctime)s) %(message)s',
        datefmt='%m-%d %H:%M',
    )