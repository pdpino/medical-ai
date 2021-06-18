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
        """heart size is normal .
        mediastinum is normal .
        lungs are clear .
        there is no pleural effusion or pneumothorax""",
    ),
]

GROUPS_mimic_v3 = [
    (
        ACTUAL_DISEASES, 0, 'no acute cardiopulmonary process',
    ),
    (
        # LABELS ARE WRONG!! should be SD=1
        ('Cardiomegaly', 'Pleural Effusion', 'Edema', 'Atelectasis', 'Support Devices'),
        (1, 1, 1, 1, 0),
        """in comparison with the study of xxxx ,
        the monitoring and support devices are unchanged .
        continued enlargement of the cardiac silhouette with
        pulmonary vascular congestion and bilateral pleural effusions
        with compressive atelectasis at the bases""",
    ),
]

# This version is only used with prefix
GROUPS_mimic_v3_woPREFIX = [
    (
        ACTUAL_DISEASES, 0, 'no acute cardiopulmonary process',
    ),
    (
        # LABELS ARE WRONG!! should be SD=1
        ('Cardiomegaly', 'Pleural Effusion', 'Edema', 'Atelectasis', 'Support Devices'),
        (1, 1, 1, 1, 0),
        """the monitoring and support devices are unchanged .
        continued enlargement of the cardiac silhouette
        with pulmonary vascular congestion and bilateral pleural effusions
        with compressive atelectasis at the bases""",
    ),
]

PREFIX_mimic = 'in comparison with the study of xxxx , '



GROUPS_mimic_v4 = [
    (
        # Constant-mimic-v4 report, achieves high bleu!
        ACTUAL_DISEASES, 0,
        """in comparison with the study of xxxx ,
there is little change and no evidence of acute cardiopulmonary disease .
the heart is normal in size . the mediastinum is unremarkable .
no pneumonia , vascular congestion , or pleural effusion""",
    ),
]

GROUPS_mimic_v5 = [
    (
        ACTUAL_DISEASES, 0, 'no acute cardiopulmonary process',
    ),
    (
        ('Cardiomegaly', 'Pleural Effusion', 'Edema', 'Atelectasis', 'Support Devices'),
        (1, 1, 1, 1, 0),
        """the monitoring and support devices are unchanged .
        continued enlargement of the cardiac silhouette
        with pulmonary vascular congestion
        and bilateral pleural effusions
        with compressive atelectasis at the bases""",
    ),
    (
        ('Cardiomegaly', 'Enlarged Cardiomediastinum'), 0,
        'the heart is normal in size . the mediastinum is unremarkable',
    ),
    (
        ('Pneumonia', 'Pleural Effusion', 'Edema'), 0,
        'no pneumonia , vascular congestion , or pleural effusion',
    ),
]
PREFIX_mimic_v2 = 'in comparison with the study of xxxx , there is little change .'


GROUPS_mimic_v6 = [
    (
        ACTUAL_DISEASES, 0,
        'no acute cardiopulmonary process',
    ),
    (
        ('Cardiomegaly', 'Pleural Effusion', 'Edema', 'Atelectasis', 'Support Devices'),
        (1, 1, 1, 1, 1),
        """in comparison with the study of xxxx ,
        the monitoring and support devices are unchanged .
        continued enlargement of the cardiac silhouette with
        pulmonary vascular congestion and bilateral pleural effusions
        with compressive atelectasis at the bases""",
    ),
    (
        ('Cardiomegaly', 'Pleural Effusion', 'Edema', 'Atelectasis', 'Support Devices'),
        (1, 1, 1, 1, 0),
        """continued enlargement of the cardiac silhouette
        with pulmonary vascular congestion
        and bilateral pleural effusions
        with compressive atelectasis at the bases""",
    ),
]

# GROUPED_mimic_debug = [
#     ('Pneumonia', 0, 'no evidence of pneumonia'),
#     (
#         ('Cardiomegaly', 'Enlarged Cardiomediastinum', *_LUNG_RELATED_DISEASES), 0,
#         """normal heart , lungs , hila , mediastinum , and pleural surfaces .
#         no evidence of intrathoracic malignancy or infection ."""
#     ),
#     (
#         _LUNG_RELATED_DISEASES, 0, 'lung volumes normal , lungs are clear',
#     ),
#     (
#         ('Cardiomegaly', 'Enlarged Cardiomediastinum'), 0,
#         'the heart is normal in size . the mediastinum is unremarkable',
#     ),
#     (
#         ('Pneumonia', 'Pleural Effusion', 'Edema'), 0,
#         'no pneumonia , vascular congestion , or pleural effusion',
#     ),
#     ('Edema', 1, 'mild pulmonary edema .'),
#     ('Atelectasis', 1, 'low lung volumes with bibasilar atelectasis .'),
# ]
