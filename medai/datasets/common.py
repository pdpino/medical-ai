from collections import namedtuple

BatchItem = namedtuple('BatchItem', [
    'image',
    'labels',
    'report',
    'filename',
])

BatchItems = namedtuple('BatchItems', [
    'images',
    'labels',
    'reports',
    'filenames',
    'stops', # Stop signals for hierarchical decoders
])

BatchItem.__new__.__defaults__ = (None,) * len(BatchItem._fields)
BatchItems.__new__.__defaults__ = (None,) * len(BatchItems._fields)

CHEXPERT_LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices',
]