from datetime import datetime
import time
import os
import numpy as np

WORKSPACE_DIR = os.environ['MED_AI_WORKSPACE_DIR']

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