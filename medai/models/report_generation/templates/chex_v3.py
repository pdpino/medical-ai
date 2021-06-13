"""Templates chex-v3.

Simple sentences but with higher NLP metric."""

TEMPLATES_CHEXPERT_v3 = {
    'Cardiomegaly': {
        0: 'the heart is normal in size',
        1: 'the heart is enlarged in size',
    },
    'Enlarged Cardiomediastinum': {
        0: 'the mediastinum is unremarkable',
        1: 'the mediastinum silhouette is enlarged',
    },
    'Lung Lesion': {
        0: 'there are no pulmonary nodules or mass lesions identified',
        1: 'there are visible pulmonary nodules or mass lesions identified',
    },
    'Lung Opacity': {
        0: 'there is no focal airspace disease',
        1: 'there is airspace disease',
    },
    'Edema': {
        0: 'there is no pulmonary edema',
        1: 'there is pulmonary edema seen',
    },
    'Consolidation': {
        0: 'there is no focal airspace consolidation',
        1: 'there is focal airspace consolidation',
    },
    'Pneumonia': {
        0: 'there is no evidence of pneumonia',
        1: 'there is evidence of pneumonia',
    },
    'Atelectasis': {
        0: 'there is no atelectasis',
        1: 'there is atelectasis',
    },
    'Pneumothorax': {
        0: 'there is no pneumothorax',
        1: 'there is a pneumothorax',
    },
    'Pleural Effusion': {
        0: 'there is no evidence of pleural effusion',
        1: 'there is a pleural effusion',
    },
    'Pleural Other': {
        0: 'there is no pleural thickening',
        1: 'there is pleural thickening present',
    },
    'Fracture': {
        0: 'there is no rib fracture',
        1: 'there is a fracture identified',
    },
    'Support Devices': {
        0: 'there is no device',
        1: 'there is a device , catheter with tip or line or other',
    },
}
