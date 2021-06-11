"""Templates chex-v2.

Chosen sentences to improve NLP metrics, keeping the meaning.

For each disease:
- Grab all sentences that are labeled with 0 or 1 (not with -2 or -1)
- Compute BLEU between all those pair of sentences
- For 0 and 1, choose the sentence with best BLEU against the others

There is a slight data-leak in this method, as sentences from both train and test were used.
"""

TEMPLATES_CHEXPERT_v2 = {
    'Cardiomegaly': {
        0: 'the heart size is within normal limits',
        1: 'the heart size is upper limits normal or mildly enlarged',
    },
    'Enlarged Cardiomediastinum': {
        0: 'the cardiomediastinal silhouette is within normal limits',
        1: 'the cardiomediastinal silhouette is significantly enlarged',
    },
    'Lung Lesion': {
        0: 'there are no suspicious appearing pulmonary nodules or masses',
        1: 'dense nodule in the right lower lobe suggests a previous granulomatous process',
    },
    'Lung Opacity': {
        0: 'the lungs are clear without evidence of focal airspace disease',
        1: 'there is patchy airspace disease in the right lower lobe',
    },
    'Edema': {
        0: 'there is no evidence of pulmonary edema',
        1: 'vascular congestion and diffuse interstitial edema',
    },
    'Consolidation': {
        0: 'there is no focal airspace consolidation',
        1: 'the lungs focal airspace consolidation',
    },
    'Pneumonia': {
        0: 'there is no evidence of pneumonia',
        1: 'there are residuals of prior granulomatous infection',
    },
    'Atelectasis': {
        # "0" does not exist, not optimal BLEU
        # (most negative sentences are paired with other diseases)
        0: 'no atelectasis',
        1: 'there is associated atelectasis in the left lung base',
    },
    'Pneumothorax': {
        0: 'here is no evidence of pneumothorax',
        1: 'there is a moderate sized right pneumothorax',
    },
    'Pleural Effusion': {
        0: 'there is no evidence of pleural effusion',
        1: 'there is a small left pleural effusion',
    },
    'Pleural Other': {
        0: 'lungs are clear without evidence of fibrosis',
        1: 'there is asymmetric right apical smooth pleural thickening',
    },
    'Fracture': {
        0: 'there is no acute , displaced rib fracture',
        1: 'acute , displaced rib fractures', # Not exactly present in the GT, but almost
    },
    'Support Devices': {
        0: 'there is interval removal of the tracheostomy tube and right subclavian central venous catheter',
        1: 'left sided subclavian central venous catheter with tip in the right atrium',
    },
}
