from medai.metrics.report_generation.abn_match.matchers import (
    AllWordsPattern,
    AnyGroupPattern,
    AllGroupsPattern,
    BodyPartStatusPattern,
)
from medai.metrics.report_generation.abn_match.labeler import AbnormalityLabeler

### Some leftover stuff:
# "density": many sentences! which abnormality do they belong?


_TEXTRAY_PATTERNS = {
    'Abnormal aorta': AllGroupsPattern(r'aorta|aortic', r'tortuosity|tortuous|ectatic|ectasia'),
    # Not observed in IU: twisted|uncoiled
    # mild tortuosity, mildly tortuous, mildly ectatic, descending/ascending,
    # "dilated slightly tortuous aorta"
    'Aortic calcification': AnyGroupPattern(
        AllGroupsPattern(r'aort|vascular|hilar', r'calcification|calcified'),
        AllGroupsPattern(r'unfolded|unfolding', 'aorta'), ## ???
        # AllGroupsPattern('atherosclerotic', 'changes'), ## ???
    ),
    'Artificial valve': AnyGroupPattern(
        'valve',
        AllGroupsPattern(r'cardiac|aortic|mitral|prostheti.', 'valve'),
    ),
    'Atelectasis': 'atelectasis',
    # characteristics: discoid basilar bibasilar focal lingula patchy left/right
    # subsegmental base minimal scattered midlung platelike passive
    # Often similar to infiltrate or scarring, e.g. "atelectasis versus scarring",
    # "diff diagnosis includes atelectasis, infiltrate, scarring"
    # Often co-seen with opacities: "there is bibasilar airspace disease, possible atelectasis",
    # "minimal opacities, subsegmental atelectasis"
    'Bronchial wall thickening': AllGroupsPattern('bronchial', r'thickening|cuffing'),
    # cuffing is referred to the same???
    'Cardiac pacer': AnyGroupPattern(
        r'pacer|pacemaker|icd',
        AllGroupsPattern('cardiac', 'generator'),
    ),
    'Cardiomegaly': AnyGroupPattern(
        'cardiomegaly',
        # AllGroupsPattern(
        #     'heart', r'large|prominen|upper|top|widen',
        # ),
        # AllGroupsPattern(
        #     'cardiac', r'contour|silhouette', r'large|prominen|upper|top|widen',
        # ),
        BodyPartStatusPattern(
            body=AnyGroupPattern(
                'heart',
                AllGroupsPattern('cardiac', r'contour|silhouette'),
            ),
            normal=r'normal',
            abnormal=r'large|prominen|upper|top|widen',
        ),
    ),
    'Central line': AnyGroupPattern(
        'picc',
        AllGroupsPattern('central', r'\bline'),
    ),
    'Consolidation': r'consolidat', # alveolar (redundant??) #  focal bibasilar
    'Costophrenic angle blunting': AllGroupsPattern('costophrenic', r'blunting|blunted'),
    'Degenerative changes': AllGroupsPattern('degenerative', 'change'),
    'Elevated diaphragm': AllGroupsPattern('elevat', 'diaphragm'),
    # Other abnormality: flattenning of the diaphragm
    # Other abnormality: diaphragm eventration
    'Fibrotic changes': 'fibrotic',
    'Fracture': AnyGroupPattern(
        'fracture',
        BodyPartStatusPattern(
            body=r'bon\w\b|osseous|skeletal',
            normal=r'intact|unremarkable|normal',
            abnormal=r'abnormalit|finding',
        ),
    ),
    'Granuloma': 'granuloma',
    'Hernia diaphragm': AnyGroupPattern(
        AllGroupsPattern('hernia', 'diaphragm|hiatal|hiatus'),
        'morgagni',
        'bochdalek',
    ),
    'Hilar prominence': AllGroupsPattern('hilar', 'contour', r'large|prominen'),
    'Hyperinflation': r'hyperinflat|hyperexpan',
    'Interstitial markings': AllGroupsPattern('interstitial', r'opacit|pattern|marking|change'),
    # coarse
    'Kyphosis': r'kyphosis|kyphotic',
    'Lung Opacity': AnyGroupPattern(
        r'opaci[ft]|infiltrat',
        AllGroupsPattern('airspace', 'disease'),
    ),
    # REVIEW: added to original text-ray labels, to compensate for missing diagnostics
    # should it be included in other label??
    'Mass': 'mass',
    'Mediastinal widening': BodyPartStatusPattern(
        body=r'cardiomediastinum|\bmediastinum|mediastinal',
        normal=r'normal|unremarkable',
        abnormal=r'large|prominen|widen',
    ),
    'Much bowel gas': AllGroupsPattern('bowel', 'gas'), # not found in IU!
    'Nodule': r'nodule|nodular',
    'Orthopedic surgery': 'orthopedic', # not found in IU
    'Osteopenia': r'osteopenia|osteophyte|osteoarthr|osteoporosi|arthriti',
    'Pleural effusion': AnyGroupPattern(
        'effusion',
        AllGroupsPattern('pleural', r'fluid|effusion'),
    ),
    'Pleural thickening': AllGroupsPattern('pleural', 'thickening'),
    'Pneumothorax': 'pneumothora',
    'Pulmonary edema': AnyGroupPattern(
        r'edema|chf',
        AllGroupsPattern('heart', 'failure'),
        AllGroupsPattern(r'pulmonar|vascular', r'congestion|prominence'),
    ),
    'Rib fracture': AllGroupsPattern('rib', 'fracture'),
    'Scoliosis': r'scoliosis|scolioti\w|(dextro|levo)curvature', # dextro, levo
    'Soft tissue calcifications': AnyGroupPattern(
        AllGroupsPattern(r'liver|lymph|chondralabdomen|cyst|epigastrium', r'calcif'),
        AllWordsPattern('soft', 'tissue', 'density'),
    ),
    'Sternotomy wires': 'sternotomy',
    'Surgical clips noted': 'clip',
    'Thickening of fissure': AllGroupsPattern('thickening', 'fissure'),
    'Trachea deviation': BodyPartStatusPattern(
        body='trachea',
        normal='midline',
        abnormal='deviation',
    ),
    'Transplant': 'transplant',
    'Tube': r'tube\b|ivc|svc|catheter', # ivc/svc/catheter is tube or line??
    'Vertebral height loss': AllGroupsPattern('narrow', r'vertebra\w'),
}


class TextRayLabeler(AbnormalityLabeler):
    name = 'textray'
    metric_name = 'abn-textray'

    patterns = _TEXTRAY_PATTERNS
    diseases = sorted(list(_TEXTRAY_PATTERNS))
