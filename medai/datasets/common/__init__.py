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

# TODO: use only BatchItem everywhere, and always use plural (images, reports, etc)
BatchRGItems = namedtuple('BatchRGItems', [
    'images',
    'labels',
    'reports',
    'report_fnames',
    'image_fnames',
    'stops', # Stop signals for hierarchical decoders
    'masks',
    'sentence_embeddings',
])

BatchItem.__new__.__defaults__ = (-1,) * len(BatchItem._fields)
BatchRGItems.__new__.__defaults__ = (None,) * len(BatchRGItems._fields)

# Organ masks
UP_TO_DATE_MASKS_VERSION = 'v2'

# Reports versions
LATEST_REPORTS_VERSION = 'v3'
