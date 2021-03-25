from functools import partial

from medai.training.detection.h2bb.thresh_largest_area import h2bb_thresh_largest_area

_H2BB_METHODS = {
    'thr-largarea': h2bb_thresh_largest_area,
}

AVAILABLE_H2BB_METHODS = list(_H2BB_METHODS)

def get_h2bb_method(name, kwargs):
    if name not in _H2BB_METHODS:
        raise Exception(f'Method not found: {name}')

    method = _H2BB_METHODS[name]

    return partial(method, **kwargs)
