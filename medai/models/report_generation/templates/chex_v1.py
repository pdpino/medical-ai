# NOTE: sentences not present in GT have been tested with chexpert

TEMPLATES_CHEXPERT_v1 = {
    'Cardiomegaly': {
        0: 'heart size is normal',
        1: 'the heart is enlarged',
    },
    'Enlarged Cardiomediastinum': {
        0: 'the mediastinal contour is normal',
        1: 'the cardiomediastinal silhouette is enlarged',
    },
    'Lung Lesion': {
        0: 'no pulmonary nodules or mass lesions identified',
        1: 'there are pulmonary nodules or mass identified', # Not present in GT sentences
    },
    'Lung Opacity': {
        0: 'the lungs are free of focal airspace disease',
        1: 'one or more airspace opacities can be seen', # Not present in GT sentences
    },
    'Edema': {
        0: 'no pulmonary edema',
        1: 'pulmonary edema is seen', # Not present in GT sentences
    },
    'Consolidation': {
        0: 'no focal consolidation',
        1: 'there is focal consolidation', # Not present in GT sentences
    },
    'Pneumonia': {
        0: 'no pneumonia',
        1: 'there is evidence of pneumonia', # Not present in GT sentences
    },
    'Atelectasis': {
        # most negative sentences are paired with other diseases
        0: 'no atelectasis', # Not present in GT sentences
        1: 'appearance suggest atelectasis',
    },
    'Pneumothorax': {
        0: 'no pneumothorax is seen',
        1: 'there is pneumothorax',
    },
    'Pleural Effusion': {
        0: 'no pleural effusion',
        1: 'pleural effusion is seen', # Not present in GT sentences
    },
    'Pleural Other': {
        0: 'no fibrosis',
        1: 'pleural thickening is present', # Not present in GT sentences
    },
    'Fracture': {
        0: 'no fracture is seen',
        1: 'a fracture is identified', # Not present in GT sentences
    },
    'Support Devices': {
        0: '', # Empty on purpose
        1: 'a device is seen', # Not present in GT sentences
    },
}



# NOTE: Wrong chexpert labeler:
# enlarged-cardiom == 1
#   'heart and mediastinum within normal limits'
#   'contour irregularity of the left clavicle appears chronic and suggests old injury'
#   'chronic appearing contour deformity of the distal right clavicle suggests old injury .'
#   'elevated right hemidiaphragm , with a nodular soft tissue contour , containing liver .',

# lung-lesion == 1
#   ct scan is more sensitive in detecting small nodules
#   no suspicious appearing lung nodules .

# lung-opacity == 1
#   no focal air space opacities .

# consolidation == 1
#   'no focal airspace consolidations .'
#   'no focal air is prominent consolidation .'

# pleural-effusion == 0
#   no knee joint effusion

# pleural-effusion == 1
#   no findings to suggest pleural effusion

# fracture==1
#   no visible fractures
#   no displaced rib fractures
#  'no acute , displaced rib fractures .',
#   'no displaced rib fracture visualized .',
#   'no definite visualized rib fractures .',
#   'no displaced rib fractures identified .',
#   'no displaced rib fractures visualized .'

# support-devices == 0
#  'no evidence of tuberculosis .',
#   'no evidence of active tuberculosis .',
# 'there is no evidence of tuberculous disease .',
# 'specifically , there is no evidence of tuberculous disease .'
