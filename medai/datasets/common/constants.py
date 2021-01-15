"""Define disease and organ constants."""

CHEXPERT_LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices',
]

CXR14_DISEASES = [
    'Atelectasis',
    'Cardiomegaly',
    'Effusion',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pneumonia',
    'Pneumothorax',
    'Consolidation',
    'Edema',
    'Emphysema',
    'Fibrosis',
    'Pleural_Thickening',
    'Hernia',
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
