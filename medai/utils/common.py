from datetime import datetime
import time
import os
import logging
import numpy as np

WORKSPACE_DIR = os.environ['MED_AI_WORKSPACE_DIR']
TMP_DIR = os.path.join(WORKSPACE_DIR, 'tmp')
CACHE_DIR = os.path.join(WORKSPACE_DIR, 'cache')


def get_timestamp(short=True):
    now = datetime.fromtimestamp(time.time())

    if short:
        return now.strftime('%m%d_%H%M%S')
    return now.strftime('%Y-%m-%d_%H-%M-%S')


def duration_to_str(all_seconds):
    all_seconds = int(all_seconds)
    seconds = int(all_seconds % 60)
    minutes = all_seconds // 60
    hours = minutes // 60

    minutes = minutes % 60

    if hours > 0:
        return f'{hours}h {minutes}m {seconds}s'
    elif minutes > 0:
        return f'{minutes}m {seconds}s'
    return f'{seconds}s'


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
    # arr shape: *, height, width

    shape = arr.size()
    flatten = arr.view(*shape[:-2], -1)
    # shape: *, height*width

    arr_min = flatten.min(-1)[0]
    # shape: *

    arr_max = flatten.max(-1)[0]
    # shape: *

    arr_min = arr_min.unsqueeze(-1).unsqueeze(-1)
    arr_max = arr_max.unsqueeze(-1).unsqueeze(-1)
    # shape: *, 1, 1

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


def divide_arrays(a, b):
    """Divide two np.arrays element-wise, avoiding NaN values in the result."""
    dont_use = b == 0

    a = a.copy()
    a[dont_use] = 0

    b = b.copy()
    b[dont_use] = 1

    return a / b


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
    except ValueError:
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


def timeit_main(logger):
    """Times a main function."""
    def wrapper(fn):
        def wrapped(*args, **kwargs):
            """Wraps the function with a timer."""
            start_time = time.time()

            result = fn(*args, **kwargs)

            total_time = time.time() - start_time
            logger.info('Total time: %s', duration_to_str(total_time))
            logger.info('=' * 50)

            return result

        return wrapped

    return wrapper
