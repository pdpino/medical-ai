"""Defines the lighter chexpert labeler.

Uses the AbnormalityLabeler logic, but with chexpert-labeler patterns.
"""

from medai.datasets.common import CHEXPERT_DISEASES
from medai.metrics.report_generation.abn_match.matchers import (
    AnyGroupPattern,
    AllGroupsPattern,
    AllWordsPattern,
)
from medai.metrics.report_generation.abn_match.labeler import AbnormalityLabeler

_CHEXPERT_PATTERNS = {
    'Enlarged Cardiomediastinum': AnyGroupPattern(
        AllGroupsPattern(r'cardiomediastinum|\bmediastinum|mediastinal', r'large|prominen|widen'),
        AllGroupsPattern('hilar', 'contour', r'large|prominen'),
    ),
    'Cardiomegaly': AnyGroupPattern(
        'cardiomegaly',
        AllGroupsPattern(
            r'heart', r'large|prominen|upper|top|widen',
        ),
        AllGroupsPattern(
            'cardiac', r'contour|silhouette', r'large|prominen|upper|top|widen',
        ),
    ),
    'Consolidation': r'consolidat',
    'Edema': AnyGroupPattern(
        r'edema|chf',
        AllWordsPattern('heart', 'failure'),
        AllGroupsPattern(r'pulmonar|vascular', r'congestion|prominence'),
    ),
    'Lung Lesion': AnyGroupPattern(
        r'mass|nodule|tumor|neoplasm|carcinoma|lump',
        AllGroupsPattern('nodular', r'densit|opaci[tf]'),
        AllWordsPattern('cavitary', 'lesion'),
    ),
    'Lung Opacity': AnyGroupPattern(
        r'opaci|infiltrate|infiltration|reticulation|scar',
        AllGroupsPattern(r'interstitial|reticular', r'marking|pattern|lung'),
        AllGroupsPattern(r'air[\s\-]*space', 'disease'),
        AllWordsPattern('parenchymal', 'scarring'),
        AllGroupsPattern(r'peribronchial|wall', 'thickening'),
    ),
    'Pneumonia': r'pneumonia|infectio',
    'Atelectasis': r'atelecta|collapse',
    'Pneumothorax': r'pneumothora',
    'Pleural Effusion': AnyGroupPattern(
        'effusion',
        AllGroupsPattern('pleural', r'fluid|effusion'),
    ),
    'Pleural Other': AnyGroupPattern(
        r'fibrosis|fibrothorax',
        AllGroupsPattern(r'pleural|pleuro\-(parenchymal|pericardial)', 'scar'),
        AllWordsPattern('pleural', 'thickening'),
    ),
    'Fracture': 'fracture',
    'Support Devices': AnyGroupPattern(
        # pylint: disable=line-too-long
        r'pacer|\bline\b|lines|picc|tube|valve|catheter|pacemaker|hardware|arthroplast|marker|icd|defib|device|drain\b|plate|screw|cannula|aparatus|coil|support|equipment|mediport',
    ),
}

class ChexpertLighterLabeler(AbnormalityLabeler):
    name = 'chexpert'
    metric_name = 'lighter-chex'

    patterns = _CHEXPERT_PATTERNS
    diseases = list(CHEXPERT_DISEASES[1:]) # Remove No Finding
