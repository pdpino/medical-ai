import os
import logging
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

from medai.datasets.common import (
    BatchItem,
    CHEXPERT_DISEASES,
)
from medai.utils.images import get_default_image_transform

LOGGER = logging.getLogger(__name__)

DATASET_DIR = os.environ.get('DATASET_DIR_CHEXPERT')

_DATASET_STATS_BY_FRONTAL_ONLY = {
    True: (0.5065, 0.2895),
    False: (0.5031, 0.2913),
}


class ChexpertDataset(Dataset):
    def __init__(self, dataset_type='train', labels=None,
                 max_samples=None,
                 image_size=(512, 512), norm_by_sample=False, image_format='RGB',
                 frontal_only=False,
                 **unused_kwargs):
        super().__init__()

        if DATASET_DIR is None:
            raise Exception('DATASET_DIR_CHEXPERT not found in env variables')

        self.dataset_type = dataset_type
        self.image_format = image_format
        self.image_size = image_size
        _DATASET_MEAN, _DATASET_STD = _DATASET_STATS_BY_FRONTAL_ONLY[frontal_only]
        self.transform = get_default_image_transform(
            self.image_size,
            norm_by_sample=norm_by_sample,
            mean=_DATASET_MEAN,
            std=_DATASET_STD,
        )

        self.image_dir = DATASET_DIR

        # Load split images
        _AVAILABLE_SPLITS = ('train', 'val')
        if dataset_type not in _AVAILABLE_SPLITS:
            raise ValueError(f'No such dataset_type, must be one of {_AVAILABLE_SPLITS}')

        # Choose diseases names
        if not labels:
            self.labels = list(CHEXPERT_DISEASES)
        else:
            # Keep only the ones that exist
            self.labels = [d for d in labels if d in CHEXPERT_DISEASES]

            not_found_diseases = list(set(labels) - set(CHEXPERT_DISEASES))
            if not_found_diseases:
                LOGGER.warning('Diseases not found: %s (ignoring)', not_found_diseases)

        self.n_diseases = len(self.labels)
        self.multilabel = True

        assert self.n_diseases > 0, 'No diseases selected!'

        # Load csv file
        if dataset_type == 'val':
            dataset_type = 'valid'
        fpath = os.path.join(DATASET_DIR, f'{dataset_type}.csv')
        self.label_index = pd.read_csv(fpath)

        # Keep only useful columns
        columns = ['Path', 'Frontal/Lateral'] + CHEXPERT_DISEASES
        self.label_index = self.label_index[columns]

        # Simplify paths
        self.label_index.replace(r'CheXpert-v1.0-small/', r'', regex=True, inplace=True)

        # Replace uncertain and absent labels
        self.label_index.replace(-1, 1, inplace=True) # Uncertainty --> treated as positive
        self.label_index.fillna(0, inplace=True) # Absent --> treated as negative

        # Keep only frontal images
        if frontal_only:
            self.label_index = self.label_index[self.label_index['Frontal/Lateral'] == 'Frontal']

        # Keep only max_samples images
        if max_samples:
            self.label_index = self.label_index[:max_samples]

        self.label_index.reset_index(drop=True, inplace=True)


    def __len__(self):
        return len(self.label_index)

    def __getitem__(self, idx):
        # Load image_name and labels
        row = self.label_index.iloc[idx]

        # Image name
        image_fname = row['Path']

        # Extract labels
        labels = row[self.labels].to_numpy().astype('int')

        # Load image
        image_fpath = os.path.join(self.image_dir, image_fname)
        try:
            image = Image.open(image_fpath).convert(self.image_format)
        except OSError as e:
            LOGGER.error(
                '%s: Failed to load image, may be broken: %s',
                self.dataset_type, image_fpath,
            )
            LOGGER.error(e)

            # FIXME: a way to ignore the image during training? (though it may break other things)
            raise

        image = self.transform(image)

        return BatchItem(
            image=image,
            labels=labels,
            image_fname=image_fname,
        )

    def get_labels_presence_for(self, target_label):
        """Returns a list of tuples (idx, 0/1) indicating presence/absence of a
            label for each sample.
        """
        if isinstance(target_label, int):
            target_label = self.labels[target_label]

        return list(self.label_index[target_label].items())

    def get_presence_for_no_finding(self):
        return self.get_labels_presence_for('No Finding')
