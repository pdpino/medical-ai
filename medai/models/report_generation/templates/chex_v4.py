""".

Difference with chex_v1: using synonyms, to test synonym stress-test
"""

from medai.models.report_generation.templates import chex_v1
from medai.datasets.common.constants import CHEXPERT_DISEASES

TEMPLATES_CHEXPERT_v4_syn = {
    'Cardiomegaly': {
        0: 'no cardiomegaly',
        1: 'cardiomegaly',
    },
    'Enlarged Cardiomediastinum': {
        0: 'cardiomediastinal silhouette is normal',
        1: 'mediastinal contour is enlarged',
    },
    'Lung Lesion': {
        0: 'no nodular opacities identified',
        1: 'there are nodular opacities identified',
    },
    'Lung Opacity': {
        0: 'no infiltrate',
        1: 'interstitial markings',
    },
    'Edema': {
        0: 'no vascular congestion',
        1: 'vascular congestion is seen',
    },
    'Consolidation': { # remains the same
        0: 'no focal consolidation',
        1: 'there is focal consolidation',
    },
    'Pneumonia': {
        0: 'no infection',
        1: 'there is evidence of infection',
    },
    'Atelectasis': {
        0: 'no collapsed lungs',
        1: 'collapsed lungs',
    },
    'Pneumothorax': { # remains the same
        0: 'no pneumothorax is seen',
        1: 'there is pneumothorax',
    },
    'Pleural Effusion': {
        0: 'no pleural fluid',
        1: 'pleural fluid is seen',
    },
    'Pleural Other': {
        0: 'no pleural scar',
        1: 'pleural scar is seen',
    },
    'Fracture': { # remains the same
        0: 'no fracture is seen',
        1: 'a fracture is identified',
    },
    'Support Devices': {
        0: '', # Empty on purpose
        1: 'a line is seen',
    },
}

TEMPLATES_CHEXPERT_v4_noisy = {
    'Cardiomegaly': {
        0: 'heart size is normal',
        1: 'the heart is mildly enlarged',
    },
    'Enlarged Cardiomediastinum': {
        0: 'the mediastinal contour is normal',
        1: 'the cardiomediastinal silhouette is prominent enlarged',
    },
    'Lung Lesion': {
        0: 'no pulmonary nodules or mass lesions identified',
        1: 'there are small right pulmonary nodules and a calcified mass in the left upper lobe',
    },
    'Lung Opacity': {
        0: 'the lungs are free of focal airspace disease',
        1: 'left upper lobe airspace opacities are seen',
    },
    'Edema': {
        0: 'no pulmonary edema',
        1: 'mild interstitial pulmonary edema is seen',
    },
    'Consolidation': {
        0: 'no focal consolidation',
        1: 'there is focal consolidation in the left lower lobe',
    },
    'Pneumonia': {
        0: 'no pneumonia',
        1: 'there is evidence of right lower lobe pneumonia',
    },
    'Atelectasis': {
        # most negative sentences are paired with other diseases
        0: 'no atelectasis',
        1: 'appearance suggest atelectasis in the left lung base',
    },
    'Pneumothorax': {
        0: 'no pneumothorax is seen',
        1: 'there is a small right pneumothorax',
    },
    'Pleural Effusion': {
        0: 'no pleural effusion',
        1: 'small right pleural effusion is seen',
    },
    'Pleural Other': {
        0: 'no fibrosis',
        1: 'mild left biapical pleural thickening is present',
    },
    'Fracture': {
        0: 'no fracture is seen',
        1: 'old left rib fracture is identified',
    },
    'Support Devices': {
        0: '', # Empty on purpose
        1: 'there is left picc line',
    },
}


# Duplicated sentence has some noise
TEMPLATES_CHEXPERT_v4_gaming_dup_noisy = {
    abn: {
        val: chex_v1.TEMPLATES_CHEXPERT_v1[abn][val] + ' . ' + \
            TEMPLATES_CHEXPERT_v4_noisy[abn][val]
        for val in (0, 1)
    }
    for abn in CHEXPERT_DISEASES[1:]
}

TEMPLATES_CHEXPERT_v4_gaming_dup_noisy_inv = {
    abn: {
        val: TEMPLATES_CHEXPERT_v4_noisy[abn][val] + ' . ' + \
             chex_v1.TEMPLATES_CHEXPERT_v1[abn][val]
        for val in (0, 1)
    }
    for abn in CHEXPERT_DISEASES[1:]
}
