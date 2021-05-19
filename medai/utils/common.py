from datetime import datetime
import functools
import numbers
import time
import os
import random
import logging
import numpy as np
import torch

WORKSPACE_DIR = os.environ['MED_AI_WORKSPACE_DIR']
TMP_DIR = os.path.join(WORKSPACE_DIR, 'tmp')
CACHE_DIR = os.path.join(WORKSPACE_DIR, 'cache')

LOGGER = logging.getLogger(__name__)

def set_seed(num):
    """Sets a seed.

    Notice this does not ensure full reproducibility in pytorch,
    see here: https://pytorch.org/docs/stable/notes/randomness.html
    """
    if num is None:
        return

    random.seed(num)
    torch.manual_seed(num)
    np.random.seed(num)


def set_seed_from_metadata(metadata):
    seed = metadata.get('seed', None)
    if seed is not None:
        set_seed(seed)
    else:
        LOGGER.warning('Seed not found in metadata')


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


def arr_to_range(arr, min_value=0, max_value=1):
    return np.interp(arr, (arr.min(), arr.max()), (min_value, max_value))


def tensor_to_range01(arr, eps=1e-8):
    # arr shape: *, height, width

    shape = arr.size()
    flatten = arr.view(*shape[:-2], -1).detach()
    # shape: *, height*width

    arr_min = flatten.min(-1)[0]
    # shape: *

    arr_max = flatten.max(-1)[0]
    # shape: *

    arr_min = arr_min.unsqueeze(-1).unsqueeze(-1)
    arr_max = arr_max.unsqueeze(-1).unsqueeze(-1)
    # shape: *, 1, 1

    arr_range = (arr_max - arr_min) + eps
    return divide_tensors(arr - arr_min, arr_range)


def divide_tensors(a, b):
    """Divide two tensors element-wise, avoiding NaN or Inf values in the result."""
    if isinstance(b, numbers.Number):
        if b == 0:
            if isinstance(a, torch.Tensor):
                return torch.zeros_like(a, requires_grad=a.requires_grad)
            return 0
        return a / b

    dont_use = b == 0

    if dont_use.any():
        zeros = torch.zeros_like(a, requires_grad=a.requires_grad)
        a = torch.where(dont_use, zeros, a)

        ones = torch.ones_like(b, requires_grad=b.requires_grad)
        b = torch.where(dont_use, ones, b)

    return a.true_divide(b)


def divide_arrays(a, b):
    """Divide two np.arrays element-wise, avoiding NaN values in the result."""
    if isinstance(b, numbers.Number):
        if b == 0:
            if isinstance(a, np.ndarray):
                return np.zeros(a.shape)
            return 0
        return a / b

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


def timeit_main(logger, sep='=', sep_times=110):
    """Times a main function."""
    def wrapper(fn):
        def wrapped(*args, **kwargs):
            """Wraps the function with a timer."""
            start_time = time.time()

            result = fn(*args, **kwargs)

            total_time = time.time() - start_time
            logger.info('Total time: %s', duration_to_str(total_time))
            logger.info(sep * sep_times)

            return result

        return wrapped

    return wrapper


def pred_and_label_to_valoration(presence, gt):
    if presence + gt == 2:
        result = 'TP'
    elif presence + gt == 0:
        result = 'TN'
    elif presence == 1:
        result = 'FP'
    else:
        result = 'FN'
    return result


def partialclass(cls, *args, **kwargs):
    """Partial method for classes.

    If only functools.partial is used, class attributes cannot be accessed
    """
    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)
    return NewCls
