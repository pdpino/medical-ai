""".

Same as v5, second attempt.
A set of minimal templates (minimal style of sentences),
and a set of verbose
"""


TEMPLATES_CHEXPERT_v6_minimal = {
    'Cardiomegaly': {
        0: 'no cardiomegaly',
        1: 'cardiomegaly',
    },
    'Enlarged Cardiomediastinum': {
        0: 'mediastinal contour is normal',
        1: 'the mediastinum is enlarged',
    },
    'Lung Lesion': {
        0: 'no pulmonary nodules',
        1: 'nodules observed',
    },
    'Lung Opacity': {
        0: 'free of focal airspace disease',
        1: 'lung opacities seen',
    },
    'Edema': {
        0: 'no pulmonary edema',
        1: 'edema',
    },
    'Consolidation': {
        0: 'no consolidation',
        1: 'consolidation',
    },
    'Pneumonia': {
        0: 'no pneumonia',
        1: 'pneumonia in the lungs',
    },
    'Atelectasis': {
        0: 'no atelectasis',
        1: 'atelectasis',
    },
    'Pneumothorax': {
        0: 'no pneumothorax',
        1: 'pneumothorax in the lung',
    },
    'Pleural Effusion': {
        0: 'no pleural effusion',
        1: 'effusion',
    },
    'Pleural Other': {
        0: 'no fibrosis',
        1: 'pleural thickening',
    },
    'Fracture': {
        0: 'no fracture',
        1: 'rib fracture',
    },
    'Support Devices': {
        0: 'there is no picc line',
        1: 'there is a picc line',
    },
}

## Only positive ones are verbose
TEMPLATES_CHEXPERT_v6_verbose = {
    'Cardiomegaly': {
        0: 'no cardiomegaly',
        1: 'the cardiac silhouette size is enlarged',
    },
    'Enlarged Cardiomediastinum': {
        0: 'mediastinal contour is normal',
        1: 'the cardiomediastinal silhouette is observed enlarged',
    },
    'Lung Lesion': {
        0: 'no pulmonary nodules',
        1: 'there are pulmonary lung nodules observed',
    },
    'Lung Opacity': {
        0: 'free of focal airspace disease',
        1: 'there are present lung airspace opacities',
    },
    'Edema': {
        0: 'no pulmonary edema',
        1: 'there is noted pulmonary edema',
    },
    'Consolidation': {
        0: 'no consolidation',
        1: 'there is observed airspace consolidation',
    },
    'Pneumonia': {
        0: 'no pneumonia',
        1: 'observed pneumonia in the lungs',
    },
    'Atelectasis': {
        0: 'no atelectasis',
        1: 'there is observed present atelectasis',
    },
    'Pneumothorax': {
        0: 'no pneumothorax',
        1: 'there is noted pneumothorax in the lung',
    },
    'Pleural Effusion': {
        0: 'no pleural effusion',
        1: 'there is an observed pleural effusion',
    },
    'Pleural Other': {
        0: 'no fibrosis',
        1: 'there is present pleural thickening',
    },
    'Fracture': {
        0: 'no fracture',
        1: 'there is a noted rib fracture',
    },
    'Support Devices': {
        0: 'there is no picc line',
        1: 'there is a noted picc line',
    },
}



## Positive are verbose and ambiguous!!
TEMPLATES_CHEXPERT_v6_verbose_amb = {
    'Cardiomegaly': {
        0: 'no cardiomegaly',
        1: 'the heart is stable , mild , moderate , severe or enlarged in size',
    },
    'Enlarged Cardiomediastinum': {
        0: 'mediastinal contour is normal',
        1: 'the cardiomediastinal silhouette is unchanged or enlarged or widened',
    },
    'Lung Lesion': {
        0: 'no pulmonary nodules',
        1: 'there are left or right pulmonary lung nodules observed',
    },
    'Lung Opacity': {
        0: 'free of focal airspace disease',
        1: 'there are left or right present lung airspace opacities',
    },
    'Edema': {
        0: 'no pulmonary edema',
        1: 'there is noted mild or moderate or severe pulmonary edema',
    },
    'Consolidation': {
        0: 'no consolidation',
        1: 'there is observed left or right lung consolidation',
    },
    'Pneumonia': {
        0: 'no pneumonia',
        1: 'observed process left or right lung pneumonia',
    },
    'Atelectasis': {
        0: 'no atelectasis',
        1: 'there is observed left or right lung present atelectasis',
    },
    'Pneumothorax': {
        0: 'no pneumothorax',
        1: 'there is noted left sided or right sided , small , moderate or large pneumothorax in the lung',
    },
    'Pleural Effusion': {
        0: 'no pleural effusion',
        1: 'there is an observed left or right or bilateral , small , moderate or large pleural effusion',
    },
    'Pleural Other': {
        0: 'no fibrosis',
        1: 'there is present left or right , minimal , mild or severe pleural thickening',
    },
    'Fracture': {
        0: 'no fracture',
        1: 'there is a rib or clavicular left or right sided fracture',
    },
    'Support Devices': {
        0: 'there is no picc line',
        1: 'there is a noted right sided or left sided picc or tube',
    },
}
