from collections import namedtuple

from medai.datasets.common.constants import *

BatchItem = namedtuple('BatchItem', [
    'image',
    'labels',
    'report',
    'image_fname',
    'report_fname',
    'bboxes',
    'bboxes_valid',
    'masks',
    'original_size',
])

# NOTE: in practice, BatchItems is only used in RG (not other tasks)
# TODO: use only BatchItem everywhere, and always use plural (images, reports, etc)
BatchItems = namedtuple('BatchItems', [
    'images',
    'labels',
    'reports',
    'report_fnames',
    'image_fnames',
    'stops', # Stop signals for hierarchical decoders
    'masks',
])

BatchItem.__new__.__defaults__ = (-1,) * len(BatchItem._fields)
BatchItems.__new__.__defaults__ = (None,) * len(BatchItems._fields)
