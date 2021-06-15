"""Templates grouped by multiple diseases.

- If a group of sentences is absent, uses group-sentences, similar to constant model.
- If a disease is present, fallback to some other templates to find sentence
    (such as chex-v1).
- Achieves better NLP than chex-v1 (more similar to constant-model), and the same chex values.
"""
from medai.datasets.common import CHEXPERT_DISEASES

ACTUAL_DISEASES = list(CHEXPERT_DISEASES[1:])

_LUNG_RELATED_DISEASES = (
    'Lung Lesion',
    'Lung Opacity',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
)
GROUPS_v1 = [
    # The first two are put here just to preserve this order
    (
        ('Cardiomegaly',),
        0, 'the heart is normal in size',
    ),
    (
        ('Enlarged Cardiomediastinum',),
        0, 'the mediastinum is unremarkable',
    ),
    (
        _LUNG_RELATED_DISEASES,
        0, 'the lungs are clear',
    ),
    (
        ('Pneumothorax', 'Pleural Effusion', 'Lung Opacity'),
        0, 'there is no pneumothorax or pleural effusion . no focal airspace disease',
    ),
    (
        ('Pneumothorax', 'Pleural Effusion'),
        0, 'no pleural effusion or pneumothorax',
    ),
]

GROUPS_v2 = [
    (
        ('Cardiomegaly', 'Enlarged Cardiomediastinum'),
        0, 'the heart size and mediastinal silhouette are within normal limits',
    ),
    (
        _LUNG_RELATED_DISEASES,
        0, 'the lungs are clear',
    ),
    (
        ('Pneumothorax', 'Pleural Effusion', 'Lung Opacity'),
        0, 'there is no pneumothorax or pleural effusion . no focal airspace disease',
    ),
]

GROUPS_mimic_v1 = [
    (
        ACTUAL_DISEASES,
        0, 'no acute cardiopulmonary process',
    ),
]

GROUPS_mimic_v2 = [
    (
        ACTUAL_DISEASES,
        0, 'no acute cardiopulmonary process',
    ),
    (
        ('Pneumonia', 'Edema', 'Pleural Effusion'),
        0, 'no pneumonia , edema , or effusion',
    ),
    (
        ('Cardiomegaly', 'Pneumothorax', 'Pleural Effusion'),
        0,
        'heart size is normal . mediastinum is normal . lungs are clear . there is no pleural effusion or pneumothorax',
    ),
]
