""".

New attempt: simplify initial templates, and then add verbosity with simple
words (seen, identified, etc)
"""

TEMPLATES_CHEXPERT_v5_clean = {
    'Cardiomegaly': {
        0: 'no cardiomegaly',
        1: 'cardiomegaly',
    },
    'Enlarged Cardiomediastinum': {
        0: 'mediastinal contour is normal',
        1: 'mediastinal contour is enlarged',
    },
    'Lung Lesion': {
        0: 'no pulmonary nodules',
        1: 'there are pulmonary nodules',
    },
    'Lung Opacity': {
        0: 'free of focal airspace disease',
        1: 'there are airspace opacities',
    },
    'Edema': {
        0: 'no pulmonary edema',
        1: 'there is a pulmonary edema',
    },
    'Consolidation': {
        0: 'no consolidation',
        1: 'there is consolidation',
    },
    'Pneumonia': {
        0: 'no pneumonia',
        1: 'there is pneumonia',
    },
    'Atelectasis': {
        0: 'no atelectasis',
        1: 'atelectasis',
    },
    'Pneumothorax': {
        0: 'no pneumothorax',
        1: 'there is a pneumothorax',
    },
    'Pleural Effusion': {
        0: 'no pleural effusion',
        1: 'there is a pleural effusion',
    },
    'Pleural Other': {
        0: 'no fibrosis',
        1: 'pleural thickening',
    },
    'Fracture': {
        0: 'no fracture',
        1: 'there is a fracture',
    },
    'Support Devices': {
        0: 'there is no picc line',
        1: 'there is a picc line',
    },
}


## Only positive ones are verbose
TEMPLATES_CHEXPERT_v5_2_verbose = {
    'Cardiomegaly': {
        0: 'no cardiomegaly',
        1: 'cardiomegaly identified',
    },
    'Enlarged Cardiomediastinum': {
        0: 'mediastinal contour is normal',
        1: 'mediastinal contour is observed enlarged',
    },
    'Lung Lesion': {
        0: 'no pulmonary nodules',
        1: 'there are pulmonary nodules observed or identified',
    },
    'Lung Opacity': {
        0: 'free of focal airspace disease',
        1: 'there are airspace opacities observed',
    },
    'Edema': {
        0: 'no pulmonary edema',
        1: 'there is a pulmonary edema identified',
    },
    'Consolidation': {
        0: 'no consolidation',
        1: 'there is consolidation observed',
    },
    'Pneumonia': {
        0: 'no pneumonia',
        1: 'there is observed evidence to suggest pneumonia',
    },
    'Atelectasis': {
        0: 'no atelectasis',
        1: 'atelectasis identified observed',
    },
    'Pneumothorax': {
        0: 'no pneumothorax',
        1: 'there is a pneumothorax observed or identified',
    },
    'Pleural Effusion': {
        0: 'no pleural effusion',
        1: 'there is a pleural effusion identified',
    },
    'Pleural Other': {
        0: 'no fibrosis',
        1: 'there is pleural thickening observed',
    },
    'Fracture': {
        0: 'no fracture',
        1: 'there is a fracture identified',
    },
    'Support Devices': {
        0: 'there is no picc line',
        1: 'there is a picc line identified',
    },
}


## All of them are verbose
TEMPLATES_CHEXPERT_v5_verbose = {
    'Cardiomegaly': {
        0: 'no cardiomegaly is seen',
        1: 'cardiomegaly identified',
    },
    'Enlarged Cardiomediastinum': {
        0: 'mediastinal contour is seen normal',
        1: 'mediastinal contour is seen enlarged',
    },
    'Lung Lesion': {
        0: 'no pulmonary nodules are observed or identified',
        1: 'there are pulmonary nodules observed or identified',
    },
    'Lung Opacity': {
        0: 'no focal airspace disease observed or identified',
        1: 'there are airspace opacities observed',
    },
    'Edema': {
        0: 'no pulmonary edema observed or identified',
        1: 'there is a pulmonary edema identified',
    },
    'Consolidation': {
        0: 'no consolidation observed or identified',
        1: 'there is consolidation observed',
    },
    'Pneumonia': {
        0: 'no pneumonia is observed or identified',
        1: 'there is observed evidence to suggest pneumonia',
    },
    'Atelectasis': {
        0: 'no atelectasis observed or identified',
        1: 'atelectasis identified observed',
    },
    'Pneumothorax': {
        0: 'no pneumothorax observed or identified',
        1: 'there is a pneumothorax observed or identified',
    },
    'Pleural Effusion': {
        0: 'no pleural effusion observed or identified',
        1: 'there is a pleural effusion identified',
    },
    'Pleural Other': {
        0: 'no fibrosis observed or identified',
        1: 'there is pleural thickening observed',
    },
    'Fracture': {
        0: 'no fracture is observed or identified',
        1: 'there is a fracture identified',
    },
    'Support Devices': {
        0: 'no picc line is observed or identified',
        1: 'there is a picc line identified',
    },
}
