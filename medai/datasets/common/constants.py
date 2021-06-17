"""Define disease and organ constants."""

CHEXPERT_LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices',
]

CHEXPERT_DISEASES = list(CHEXPERT_LABELS) # Alias for the same list

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

CXR14_POS_WEIGHTS = [
    9,
    50,
    9,
    5,
    20,
    17,
    98,
    31,
    29,
    61,
    57,
    67,
    37,
    600,
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


VINBIG_DISEASES = [
    'Aortic enlargement',
    'Atelectasis',
    'Calcification',
    'Cardiomegaly',
    'Consolidation',
    'ILD',
    'Infiltration',
    'Lung Opacity',
    'Nodule/Mass',
    'Other lesion',
    'Pleural effusion',
    'Pleural thickening',
    'Pneumothorax',
    'Pulmonary fibrosis',
    'No finding',
]


# 210 labels used to train CoAtt model
COATT_LABELS = [
    'cardiac monitor', 'lymphatic diseases', 'pulmonary disease', 'osteophytes', 'foreign body',
    'dish', 'aorta, thoracic', 'atherosclerosis', 'histoplasmosis', 'hypoventilation',
    'catheterization, central venous', 'pleural effusions', 'pleural effusion', 'callus',
    'sternotomy', 'lymph nodes', 'tortuous aorta', 'stent', 'interstitial pulmonary edema',
    'cholecystectomies', 'neoplasm', 'central venous catheter', 'pneumothorax',
    'metastatic disease', 'vena cava, superior', 'cholecystectomy', 'scoliosis',
    'subcutaneous emphysema', 'thoracolumbar scoliosis', 'spinal osteophytosis',
    'pulmonary fibroses', 'rib fractures', 'sarcoidosis', 'eventration', 'fibrosis', 'spine',
    'obstructive lung disease', 'pneumonitis', 'osteopenia', 'air trapping', 'demineralization',
    'mass lesion', 'pulmonary hypertension', 'pleural diseases', 'pleural thickening',
    'calcifications of the aorta', 'calcinosis', 'cystic fibrosis', 'empyema', 'catheter',
    'lymph', 'pericardial effusion', 'lung cancer', 'rib fracture', 'granulomatous disease',
    'chronic obstructive pulmonary disease', 'rib', 'clip', 'aortic ectasia', 'shoulder',
    'scarring', 'scleroses', 'adenopathy', 'emphysemas', 'pneumonectomy', 'infection',
    'aspiration', 'bilateral pleural effusion', 'bulla', 'lumbar vertebrae', 'lung neoplasms',
    'lymphadenopathy', 'hyperexpansion', 'ectasia', 'bronchiectasis', 'nodule', 'pneumonia',
    'right-sided pleural effusion', 'osteoarthritis', 'thoracic spondylosis', 'picc',
    'cervical fusion', 'tracheostomies', 'fusion', 'thoracic vertebrae', 'catheters',
    'emphysema', 'trachea', 'surgery', 'cervical spine fusion', 'hypertension, pulmonary',
    'pneumoperitoneum', 'scar', 'atheroscleroses', 'aortic calcifications', 'volume overload',
    'right upper lobe pneumonia', 'apical granuloma', 'diaphragms', 'copd', 'kyphoses',
    'spinal fractures', 'fracture', 'clavicle', 'focal atelectasis', 'collapse',
    'thoracotomies', 'congestive heart failure', 'calcified lymph nodes', 'edema',
    'degenerative disc diseases', 'cervical vertebrae', 'diaphragm', 'humerus', 'heart failure',
    'normal', 'coronary artery bypass', 'pulmonary atelectasis', 'lung diseases, interstitial',
    'pulmonary disease, chronic obstructive', 'opacity', 'deformity', 'chronic disease',
    'pleura', 'aorta', 'tuberculoses', 'hiatal hernia', 'scolioses', 'pleural fluid',
    'malignancy', 'kyphosis', 'bronchiectases', 'congestion', 'discoid atelectasis', 'nipple',
    'bronchitis', 'pulmonary artery', 'cardiomegaly', 'thoracic aorta', 'arthritic changes',
    'pulmonary edema', 'vascular calcification', 'sclerotic', 'central venous catheters',
    'catheterization', 'hydropneumothorax', 'aortic valve', 'hyperinflation', 'prostheses',
    'pacemaker, artificial', 'bypass grafts', 'pulmonary fibrosis', 'multiple myeloma',
    'postoperative period', 'cabg', 'right lower lobe pneumonia', 'granuloma',
    'degenerative change', 'atelectasis', 'inflammation', 'effusion', 'cicatrix',
    'tracheostomy', 'aortic diseases', 'sarcoidoses', 'granulomas', 'interstitial lung disease',
    'infiltrates', 'displaced fractures', 'chronic lung disease', 'picc line',
    'intubation, gastrointestinal', 'lung diseases', 'multiple pulmonary nodules',
    'intervertebral disc degeneration', 'pulmonary emphysema', 'spine curvature', 'fibroses',
    'chronic granulomatous disease', 'degenerative disease', 'atelectases', 'ribs',
    'pulmonary arterial hypertension', 'edemas', 'pectus excavatum', 'lung granuloma',
    'plate-like atelectasis', 'enlarged heart', 'hilar calcification', 'heart valve prosthesis',
    'tuberculosis', 'old injury', 'patchy atelectasis', 'histoplasmoses', 'exostoses',
    'mastectomies', 'right atrium', 'large hiatal hernia', 'hernia, hiatal', 'aortic aneurysm',
    'lobectomy', 'spinal fusion', 'spondylosis', 'ascending aorta', 'granulomatous infection',
    'fractures, bone', 'calcified granuloma', 'degenerative joint disease',
    'intubation, intratracheal', 'others'
]
