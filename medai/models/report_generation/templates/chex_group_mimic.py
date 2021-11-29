"""Templates grouped by multiple diseases, only for MIMIC.

- This file was created later than the `chest_group.py`,
to choose better templates than that one (for MIMIC).
"""

# pylint: disable=line-too-long
GROUPS_v7 = [
    (
        (
            'Consolidation', 'Pleural Effusion', 'Pneumothorax',
            'Fracture', 'Enlarged Cardiomediastinum',
        ), 0,
        'pa and lateral views of the chest provided . there is no focal consolidation , effusion , or pneumothorax . the cardiomediastinal silhouette is normal . imaged osseous structures are intact . no free air below the right hemidiaphragm is seen .'
    ),
    (
        ('Edema', 'Pneumonia', 'Pleural Effusion'), 0,
        'in comparison with the study of xxxx , there is little change and no evidence of acute cardiopulmonary disease . no pneumonia , vascular congestion , or pleural effusion .'
    ),
    (
        ('Enlarged Cardiomediastinum', 'Pneumonia'), 0,
        'pa and lateral chest compared to xxxx normal heart , lungs , hila , mediastinum and pleural surfaces . no evidence of intrathoracic malignancy or infection .',
    ),
]

PREFIX = 'pa and lateral views of the chest provided .'

CHEX_mimic_single = {
    'Cardiomegaly': {
        0: 'heart size is normal',
        # 1: 'the heart is enlarged',
        1: 'moderate cardiomegaly is stable .',
    },
    'Enlarged Cardiomediastinum': {
        0: 'the cardiomediastinal silhouette is normal .',
        1: 'cardiomediastinal silhouette is stable .',
    },
    'Lung Lesion': {
        0: 'no lung nodules or masses',
        1: 'multiple bilateral lung nodules are again demonstrated .',
    },
    'Lung Opacity': {
        # 0: 'the lungs are free of focal airspace disease',
        0: 'no parenchymal opacities',
        # 1: 'one or more airspace opacities are seen',
        1: 'there is a persistent left retrocardiac opacity .'
    },
    'Edema': {
        0: 'there is no pulmonary edema .',
        # 1: 'pulmonary edema is seen',
        1: 'there is mild pulmonary vascular congestion .',
    },
    'Consolidation': {
        0: 'the lungs are clear without focal consolidation',
        1: 'underlying consolidation cannot be excluded',
    },
    'Pneumonia': {
        # 0: 'no evidence of intrathoracic malignancy or infection',
        0: 'no pneumonia',
        # 1: 'there is evidence of pneumonia',
        1: 'in the appropriate clinical setting , superimposed pneumonia could be considered .',
    },
    'Atelectasis': {
        0: 'no atelectasis',
        # 1: 'appearance suggest atelectasis',
        1: 'there is bibasilar atelectasis .',
    },
    'Pneumothorax': {
        0: 'there is no pneumothorax',
        1: 'there is a small left apical pneumothorax .',
    },
    'Pleural Effusion': {
        0: 'no pleural effusions .',
        # 1: 'pleural effusion is seen',
        1: 'there are small bilateral pleural effusions .',
    },
    'Pleural Other': {
        0: 'there is no evidence of fibrosis',
        # 1: 'pleural thickening is unchanged .',
        # 1: 'patient has known fibrosis in the right lower lobe .',
        # 1: 'pulmonary fibrosis is severe .',
        1: 'there is biapical pleural thickening .',
    },
    'Fracture': {
        0: 'no displaced fracture is seen',
        # 1: 'imaged osseous structures are intact .',
        1: 'multiple bilateral rib fractures are again noted .',
    },
    'Support Devices': {
        0: '', # Empty on purpose
        # 1: 'the other monitoring and support devices are constant',
        # 1: 'ng tube tip is in the stomach .',
        1: 'tracheostomy tube in standard placement .',
    },
}


# from medai.datasets.common.constants import CHEXPERT_DISEASES

## DOES NOT IMPROVE:
# (
#     CHEXPERT_DISEASES[1:],
#     0, 'there is no acute cardiopulmonary process',
# ),

_LUNG_RELATED_DISEASES = (
    'Lung Lesion',
    'Lung Opacity',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
)

GROUPS_v8 = [
    (
        _LUNG_RELATED_DISEASES, 0, 'the lungs are clear',
    ),
    (
        ('Consolidation', 'Pleural Effusion', 'Pneumothorax'), 0,
        'there is no focal consolidation , pleural effusion , or pneumothorax .',
        # { 'redundant': False },
    ),
    (
        ('Pneumothorax', 'Pleural Effusion'), 0,
        'there is no pleural effusion or pneumothorax .',
        { 'redundant': False },
    ),
    (
        ('Pneumothorax', 'Consolidation'), 0,
        'there is no focal consolidation or pneumothorax .',
        { 'redundant': False },
    ),
]


GROUPS_v9 = [
    # *GROUPS_v7,
    # (
    #     ('Lung Opacity', 'Atelectasis', 'Pleural Effusion', 'Pneumonia'), 1,
    #     'continued opacification at the base is consistent with pleural effusion and compressive atelectasis . in the appropriate clinical setting , superimposed pneumonia could be considered .',
    # ),
    # (
    #     ('Lung Opacity', 'Pneumonia', 'Atelectasis', 'Pleural Effusion'), (1, 0, 1, 1),
    #     'continued opacification at the base is consistent with pleural effusion and compressive atelectasis .',
    # ),
    # (
    #     ('Cardiomegaly', 'Enlarged Cardiomediastinum'), 0,
    #     'the heart size and mediastinum are unremarkable .',
    #     # 'the heart is of normal size with normal cardiomediastinal contours .',
    # ),
    (
        ('Edema', 'Pleural Effusion', 'Pneumothorax'), (1, 0, 0),
        'mild pulmonary edema noted no pleural effusion or pneumothorax .',
    ),
    (
        ('Cardiomegaly', 'Edema', 'Pneumonia', 'Pleural Effusion'), 0,
        'the heart is normal in size and there is no vascular congestion , pleural effusion , or acute focal pneumonia .',
    ),
    (
        ('Pneumothorax', 'Edema', 'Pleural Effusion'), 0,
        'there is no pneumothorax , vascular congestion , or pleural effusion .'
    ),
    (
        ('Edema', 'Pleural Effusion', 'Pneumonia'), 0,
        'no pneumonia , vascular congestion , or pleural effusion .',
        # 'no evidence of acute pneumonia , vascular congestion , or pleural effusion .'
        # 'no evidence of vascular congestion , pleural effusion , or acute focal pneumonia .'
        # 'in comparison with the study of xxxx , there is little change and no evidence of acute cardiopulmonary disease . no pneumonia , vascular congestion , or pleural effusion .'
    ),
    (
        ('Edema', 'Consolidation', 'Pleural Effusion'), 0,
        'there is no pulmonary edema , consolidation , or pleural effusion .',
    ),
    (
        ('Cardiomegaly', 'Edema', 'Atelectasis'), 1,
        'cardiomegaly with retrocardiac atelectasis and pulmonary edema .',
    ),
    (
        ('Lung Opacity', 'Pleural Effusion'), 0,
        'the lungs are clear without infiltrate or effusion',
    ),
    (
        ('Edema', 'Pneumonia'), 0,
        'there is no evidence of pulmonary vascular congestion or acute focal pneumonia .',
    ),

    (
        ('Cardiomegaly', 'Enlarged Cardiomediastinum'), 1,
        'heart size and mediastinum are stable .',
    ),

    (
        ('Cardiomegaly', 'Edema'), 1, 'cardiomegaly with pulmonary edema',
    ),
    (
        ('Atelectasis', 'Pleural Effusion'), 1,
        'there is a pleural effusion with atelectasis',
    ),
    (
        ('Lung Opacity', 'Pleural Effusion'), 1,
        'opacification is consistent with pleural effusion and volume in the lobe .',
    ),
    (
        ('Lung Opacity', 'Atelectasis'), 1,
        'opacities in the lung bases likely reflect atelectasis .',
    ),
    (
        ('Atelectasis', 'Pneumonia'), 1,
        'although this could reflect atelectasis , in the appropriate clinical setting superimposed pneumonia could be considered .',
    ),
    (
        ('Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pleural Effusion'),
        (1, 0, 0, 1, 1, 1),
        'there are bilateral effusions , with associated airspace disease , which most likely reflects compressive atelectasis , although pneumonia cannot be entirely excluded .',
    ),
    (
        ('Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pleural Effusion'),
        (1, 1, 0, 0, 1, 1),
        'lower lobe opacities consistent with atelectasis , edema and pleural effusion .',
    ),
    (
        ('Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pleural Effusion'),
        (1, 0, 0, 0, 1, 1),
        'there is increased opacification involving the lung , likely atelectasis and effusion .'
        # 'continued opacification at the base is consistent with pleural effusion and compressive atelectasis .',
    ),
    (
        ('Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pleural Effusion'),
        (1, 0, 0, 1, 1, 0),
        'there are opacities at the base , which likely represent atelectasis , but an underlying lower lobe pneumonia cannot be excluded .',
    ),
    (
        ('Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pleural Effusion'),
        (1, 0, 0, 1, 0, 0),
        # 'there is a new lung opacity , which is concerning for pneumonia .',
        'there are new focal opacities concerning for pneumonia',
    ),
    (
        ('Cardiomegaly', 'Edema', 'Pneumonia', 'Pleural Effusion'), (1, 0, 0, 0),
        'there is again some enlargement of the cardiac silhouette without definite vascular congestion , pleural effusion , or acute focal pneumonia .',
    ),
    # (
    #     (
    #         'Consolidation', 'Pleural Effusion', 'Pneumothorax',
    #         'Fracture', 'Enlarged Cardiomediastinum',
    #     ), 0,
    #     'pa and lateral views of the chest provided . there is no focal consolidation , effusion , or pneumothorax . the cardiomediastinal silhouette is normal . imaged osseous structures are intact . no free air below the right hemidiaphragm is seen .'
    # ),

    # (
    #     ('Support Devices', 'Pneumothorax'),
    #     (1, 0),
    #     'as compared to the previous radiograph , the patient has received a nasogastric tube . the course of the tube is unremarkable , the tip of the tube projects over the middle parts of the stomach . no complications , no pneumothorax . the other monitoring and support devices are constant .'
    # ),
    # (
    #     ('Cardiomegaly', 'Enlarged Cardiomediastinum', 'Pneumothorax', 'Pleural Effusion'),
    #     (1, 1, 0, 0),
    #     'the heart is enlarged . the mediastinal and hilar contours appear unchanged . there is no pleural effusion or pneumothorax . the lungs appear clear .'
    # ),
    # (
    #     ('Cardiomegaly', 'Enlarged Cardiomediastinum', 'Edema', 'Consolidation', 'Pleural Effusion', 'Pneumothorax'),
    #     (1, 0, 0, 0, 0, 0),
    #     'the lungs are clear without focal consolidation . no pleural effusion or pneumothorax is seen . the cardiac silhouette is enlarged . mediastinal contours are unremarkable . no pulmonary edema is seen .',
    # ),

    # (
    #     ('Enlarged Cardiomediastinum', 'Consolidation', 'Pneumothorax', 'Pleural Effusion'), 0,
    #     'pa and lateral chest radiographs were obtained . the lungs are well expanded and clear . there is no focal consolidation , effusion , or pneumothorax . the cardiomediastinal silhouette is normal .'
    # ),
    # (
    #     ('Enlarged Cardiomediastinum', 'Pneumothorax', 'Pleural Effusion'), 0,
    #     'frontal and lateral views of the chest were obtained . the lungs are clear without focal consolidation . no pleural effusion or pneumothorax is seen . the cardiac and mediastinal silhouettes are unremarkable .',
    # ),

]

# (
#     ('Cardiomegaly', 'Enlarged Cardiomediastinum', 'Edema', 'Consolidation', 'Pleural Effusion', 'Pneumothorax'),
#     (0, 0, 0, 0, 0, 0),
#     'the lungs are clear without focal consolidation . no pleural effusion or pneumothorax is seen . the cardiac silhouette is normal . mediastinal contours are unremarkable . no pulmonary edema is seen .',
# ),
