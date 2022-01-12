"""Defines the lighter chexpert labeler.

Declares the AbnormalityLabeler schema to emulate the chexpert-labeler.
"""

from medai.datasets.common import CHEXPERT_DISEASES
from medai.metrics.report_generation.abn_match.matchers import (
    AnyGroupPattern,
    AllGroupsPattern,
    AllWordsPattern,
    BodyPartStatusPattern,
)
from medai.metrics.report_generation.abn_match.labeler import AbnormalityLabeler

_CHEXPERT_PATTERNS = {
    'Enlarged Cardiomediastinum': AnyGroupPattern(
        # missing: "mild to moderate enlargement of the cardiac silhouette is unchanged"
        AllGroupsPattern(
            r'cardiomediastinum|\bmediastinum|mediastinal',
            r'large|prominen|widen|\bremarkable',
        ),
        AllGroupsPattern('hilar', 'contour', r'large|prominen'),
    ),
    'Cardiomegaly': AnyGroupPattern(
        'cardiomegaly',
        BodyPartStatusPattern(
            body=AnyGroupPattern(
                'heart',
                AllGroupsPattern('cardiac', r'contour|silhouette'),
            ),
            normal=r'normal',
            abnormal=r'large|prominen|upper|top|widen',
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
        AllGroupsPattern(r'\bair\b', r'\bspace\b', 'disease'),
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
    # NOTE: Support Devices and maybe Fracture?
    # should ignore "stable" as an uncertainty pattern
}

class ChexpertLighterLabeler(AbnormalityLabeler):
    name = 'chexpert'
    metric_name = 'lighter-chex'

    patterns = _CHEXPERT_PATTERNS
    diseases = list(CHEXPERT_DISEASES[1:]) # Remove No Finding

    lung_diseases = list(range(2, 11))
