"""Provides functions to label sentences with (JSRT) organs.

It uses multiple regexes to achieve this.
"""
import os
import re
from collections import defaultdict
from tqdm.auto import tqdm

import pandas as pd

from medai.datasets.common.constants import (
    ORGAN_BACKGROUND,
    ORGAN_HEART,
    ORGAN_LEFT_LUNG,
    ORGAN_RIGHT_LUNG,
)

### MIRQI phrases are used, so we don't start from scratch
MIRQI_DIR = os.path.join(os.environ['HOME'], 'software/MIRQI/predefined/phrases')

def load_phrases(label, mention=True):
    mention = 'mention' if mention else 'unmention'

    fname = os.path.join(MIRQI_DIR, mention, f'{label}.txt')
    with open(fname, 'r') as f:
        lines = [l.strip().replace('_', ' ') for l in f.readlines()]

    return lines


#### HEART RELATED ####
HEART_MENTIONS = \
    load_phrases('cardiomegaly') + \
    load_phrases('enlarged_cardiomediastinum') + \
    ['heart failure', 'chf', 'vascular congestion', 'vascular prominence'] + \
    [
        'heart', 'aorta', 'aortic', ' cardio', 'mediastinal', 'mediastinum', 'vascular',
        'vasculature', 'vascularity', 'cardiac',
    ]

REGEX_HEART = re.compile('|'.join(HEART_MENTIONS))

def regex_mentions_heart(sentence):
    return int(bool(REGEX_HEART.search(sentence)))


#### LUNGS RELATED ####

_lung_diseases = [
    'airspace_disease', 'airspace_opacity', 'atelectasis', 'calcinosis',
    'consolidation', 'emphysema', 'hypoinflation', 'lung_lesion',
    'pleural_effusion', 'pleural_other',
    'pneumonia', 'pneumothorax',
]
LUNGS_MENTIONS = [
    ph
    for disease in _lung_diseases
    for ph in load_phrases(disease)
]
# Edema lung related
LUNGS_MENTIONS += [
    'edema', 'pulmonary congestion',
    'clear lung',
    'the lung',
    'pleural space',
    'pleural air collection',
    'midlung',
    'alveolar',
    'pulmonary',
    'pleural',
    'costophrenic',
]


_LUNG_PATTERN_BOTH = re.compile(r'both|bilateral')
_LUNG_PATTERN_RIGHT = re.compile('right')
_LUNG_PATTERN_LEFT = re.compile('left')

REGEXES_LUNGS = [
    re.compile(r'lungs?\s(are\s)?clear'),
    re.compile(r'(right|left) lung'),
    re.compile(r'\Alung'),
    re.compile(r'lungs? volume'),
    re.compile(r'pulmon\w*\s(vascul\w*)?'),
    re.compile('expanded lungs?'),
    re.compile('(right|left) (upper |lower )?lobe'),
    re.compile('|'.join(LUNGS_MENTIONS)),
]
def regex_mentions_lungs(sentence):
    any_lung = any(pattern.search(sentence) for pattern in REGEXES_LUNGS)

    if not any_lung:
        return 0, 0

    if _LUNG_PATTERN_BOTH.search(sentence):
        return 1, 1

    left = _LUNG_PATTERN_LEFT.search(sentence)
    right = _LUNG_PATTERN_RIGHT.search(sentence)

    if not right and not left:
        # None found ("both", "right", "left")
        return 1, 1

    return int(bool(left)), int(bool(right))


#### INTRA THORACIC ####
INTRA_MENTIONS = [
    'intrathoracic',
    'central vascular',
    'cardiopulmonary',
    'bronchovascular',
    'thorax',
    'thoracic',
    'rib',
    'chest',
    'sternotomy',
]

REGEX_INTHRATORACIC = re.compile('|'.join(INTRA_MENTIONS))
def regex_mentions_intrathoracic(sentence):
    return int(bool(REGEX_INTHRATORACIC.search(sentence)))



#### OTHER DISEASES ####

_other_diseases = ['scoliosis', 'support_devices', 'fracture']

OTHER_MENTIONS = [ph for disease in _other_diseases for ph in load_phrases(disease)]

OTHER_MENTIONS += [
    'bony', 'bone', 'spine', 'osseous', 'osseus', 'skeletal', 'spondylosis', 'trachea',
    # Other support devices
    'ivc', 'clips',
]

REGEX_OTHER = re.compile('|'.join(OTHER_MENTIONS))
def regex_mentions_other(sentence):
    return int(bool(REGEX_OTHER.search(sentence)))



#### OTHER FINDINGS FROM MIRQI ####
# TODO: review each phrase (i.e. google + see some sentences) and determine to which
# organ(s) is related
# NOT USED FOR NOW

# phrases = load_phrases('other_finding')
# phrases = [
#     'blunt', # Lungs
#     'elevation',  # hemidiaphragm elevation
#     'bronchospasm', # None
#     'asthma', # None
#     'interstitial markings', # Lungs
#     'plaque', # Lungs
#     'osteophytosis', # None
#     'aortic disease', # heart or lungs
#     'bronchiolitis', # Lungs
#     'thickening',
#     'cephalization',
#     'aspiration',
#     'bullae',
#     'contusion',
#     'atherosclero',
#     'osteopenia',
#     'metastasis',
#     'granuloma',
#     'pneumomediastinum',
#     'pneumoperitoneum',
#     'osteodystrophy',
#     'cuffing',
#     'irregular lucency',
#     'inflam',
#     'fissure',
#     'prominen',
#     'kyphosis',
#     'defib',
#     'bullet',
#     'reticula',
#     'thoracentesis',
#     'bronchitis',
#     'volume loss',
#     'deformity',
#     'hemorrhage',
#     'hematoma',
#     'radiopaque',
#     'aerophagia',
#     'arthropathy',
#     'tracheostomy',
# ]



def _find_organs_for_sentence(sentence, warnings=None):
    if warnings is None:
        warnings = defaultdict(list)

    background = heart = right_lung = left_lung = 0

    if regex_mentions_other(sentence):
        background = heart = right_lung = left_lung = 1
    else:
        heart = regex_mentions_heart(sentence)
        left_lung, right_lung = regex_mentions_lungs(sentence)

    if background + heart + right_lung + left_lung == 0:
        if regex_mentions_intrathoracic(sentence):
            heart = left_lung = right_lung = 1
        else:
            warnings['all-empty'].append(sentence)
            # If nothing is identified, set all to 1
            background = heart = right_lung = left_lung = 1

    return (background, heart, right_lung, left_lung)


def find_organs_for_sentences(sentences, show=False):
    """Get organs for a collections of sentences.

    Args:
        sentences -- iterator of sentences (each as str)
    Returns:
        organs, warnings
        organs -- list of size n_sentences, with tuples (background, heart, right_lung, left_lung)
        warnings -- defaultdict(list) with problematic sentences
    """
    warnings = defaultdict(list)

    if show:
        sentences = tqdm(sentences)

    organs = [
        _find_organs_for_sentence(sentence, warnings)
        for sentence in sentences
    ]

    return organs, warnings


def save_sentences_with_organs(dataset_dir, sentences, show=False, ignore_all_ones=True):
    # Compute organs
    organs, errors = find_organs_for_sentences(sentences, show=show)

    # In the order defined in the above function
    organ_names = [ORGAN_BACKGROUND, ORGAN_HEART, ORGAN_LEFT_LUNG, ORGAN_RIGHT_LUNG]

    # Create DF
    df_organs = pd.DataFrame(organs, columns=organ_names)
    df_organs['sentence'] = sentences

    # REVIEW: add main_organ column??

    if ignore_all_ones:
        df_organs = df_organs.loc[(df_organs[ORGAN_BACKGROUND] == 0) | \
            (df_organs[ORGAN_HEART] == 0) | \
            (df_organs[ORGAN_LEFT_LUNG] == 0) | \
            (df_organs[ORGAN_RIGHT_LUNG] == 0)]

    # Save to file
    fpath = os.path.join(dataset_dir, 'reports', 'sentences_with_organs.csv')
    df_organs.to_csv(fpath, index=False)

    print(f'Sentence-2-organs saved to {fpath}')

    return df_organs, errors


MAIN_ORGANS = ['heart', 'lungs', 'thorax', 'neutral', 'all']

def get_main_organ(one_hot, sentence, warnings=None):
    """Given a one-hot array indicating organs, return the main organ indicated."""
    background, heart, right_lung, left_lung = one_hot

    neutral_sentences = warnings['all-empty'] if warnings is not None else set()

    if background == 1:
        if sentence in neutral_sentences:
            return 'neutral'
        return 'all'
    if heart + right_lung + left_lung == 3:
        return 'thorax'
    if right_lung + left_lung >= 1:
        return 'lungs'
    if heart:
        return 'heart'
    print('Case not covered: ', one_hot, sentence)
    return 'unk'
