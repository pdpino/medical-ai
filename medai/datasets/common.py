from collections import namedtuple

BatchItem = namedtuple('BatchItem', [
    'image',
    'labels',
    'report',
    'image_fname',
    'report_fname',
    'bboxes',
    'bboxes_valid',
    'masks',
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

CHEXPERT_LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices',
]

ORGAN_BACKGROUND = 'background'
ORGAN_HEART = 'heart'
ORGAN_RIGHT_LUNG = 'right lung'
ORGAN_LEFT_LUNG = 'left lung'

JSRT_ORGANS = [
    ORGAN_BACKGROUND,
    ORGAN_HEART,
    ORGAN_RIGHT_LUNG,
    ORGAN_LEFT_LUNG,
    ## Not used for now:
    # 'right clavicle',
    # 'left clavicle',
]