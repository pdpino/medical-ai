"""Templates grouped by multiple diseases.

- If a group of sentences is absent, uses group-sentences, similar to constant model.
- If a disease is present, fallback to some other templates to find sentence
    (such as chex-v1).
- Achieves better NLP than chex-v1 (more similar to constant-model), and the same chex values.
"""

_LUNG_RELATED_DISEASES = (
    'Lung Lesion',
    'Lung Opacity',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
)
GROUPS_v1 = [
    (
        ('Cardiomegaly',),
        0, 'the heart is normal in size',
    ),
    (
        ('Enlarged Cardiomediastinum',),
        0, 'the mediastinum is unremarkable',
    ),
    (
        _LUNG_RELATED_DISEASES,
        0, 'the lungs are clear',
    ),
    (
        ('Pneumothorax', 'Pleural Effusion', 'Lung Opacity'),
        0, 'there is no pneumothorax or pleural effusion . no focal airspace disease',
    ),
    (
        ('Pneumothorax', 'Pleural Effusion'),
        0, 'no pleural effusion or pneumothorax',
    ),
]
