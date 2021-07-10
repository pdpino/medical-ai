import os
import logging
import json
import pandas as pd
from torch.utils.data import Dataset

from medai.datasets.common import (
    BatchItem,
)
from medai.utils.images import (
    get_default_image_transform,
    load_image,
)

LOGGER = logging.getLogger(__name__)

DATASET_DIR = os.environ.get('DATASET_DIR_COVID_SIIM')

_DATASET_MEAN = 0.5234
_DATASET_STD = 0.2099

_BROKEN_IMAGES = set()

COVID_SIIM_DISEASES = [
    'Negative for Pneumonia',
    'Typical Appearance',
    'Indeterminate Appearance',
    'Atypical Appearance',
]

def scale_bboxes(bboxes, original_size, target_size):
    if original_size == target_size:
        return bboxes

    original_height, original_width = original_size
    height, width = target_size
    vertical_scale = height / original_height
    horizontal_scale = width / original_width

    scaled_bboxes = []
    for bbox in bboxes:
        scaled_bbox = {
            'x': horizontal_scale * bbox['x'],
            'y': vertical_scale * bbox['y'],
            'width': horizontal_scale * bbox['width'],
            'height': vertical_scale * bbox['height'],
        }

        scaled_bboxes.append(scaled_bbox)
    return scaled_bboxes


class CovidSiimDataset(Dataset):
    dataset_dir = DATASET_DIR

    def __init__(self, dataset_type='train', labels=None,
                 max_samples=None,
                 image_size=(256, 256), norm_by_sample=False, image_format='RGB',
                 **unused_kwargs):
        super().__init__()

        if DATASET_DIR is None:
            raise Exception('DATASET_DIR_COVID_SIIM not found in env variables')

        self.dataset_type = dataset_type
        self.image_format = image_format
        self.image_size = image_size
        self.transform = get_default_image_transform(
            self.image_size,
            norm_by_sample=norm_by_sample,
            mean=_DATASET_MEAN,
            std=_DATASET_STD,
        )

        self.image_dir = os.path.join(DATASET_DIR, 'images-256')

        # Choose diseases names
        if not labels:
            self.labels = COVID_SIIM_DISEASES
        else:
            # Keep only the ones that exist
            self.labels = [d for d in labels if d in COVID_SIIM_DISEASES]

            not_found_diseases = list(set(labels) - set(COVID_SIIM_DISEASES))
            if not_found_diseases:
                LOGGER.warning('Diseases not found: %s (ignoring)', not_found_diseases)

        self.n_diseases = len(self.labels)
        self.multilabel = False # CL-multilabel

        assert self.n_diseases > 0, 'No diseases selected!'

        self.master_df = pd.read_csv(os.path.join(DATASET_DIR, 'metadata.csv'))

        # Load split images
        split_fpath = os.path.join(DATASET_DIR, 'splits', f'{dataset_type}.txt')
        if not os.path.isfile(split_fpath):
            _AVAILABLE_SPLITS = os.listdir(os.path.join(DATASET_DIR, 'splits'))
            raise ValueError(f'No such dataset_type, must be one of {_AVAILABLE_SPLITS}')
        with open(split_fpath, 'r') as f:
            split_images = set(l.strip() for l in f.readlines())

        # Keep only split-images
        self.master_df = self.master_df.loc[self.master_df['study_id'].isin(split_images)]

        # Ignore broken images
        # self.master_df = self.master_df.loc[~self.master_df['image_fpath'].isin(_BROKEN_IMAGES)]

        # Keep only max_samples images
        if max_samples:
            self.master_df = self.master_df[:max_samples]

        self.master_df.reset_index(drop=True, inplace=True)

        # Load bboxes in a simpler format
        self.use_bboxes = True
        if self.use_bboxes:
            with open(os.path.join(DATASET_DIR, 'bboxes.json'), 'r') as f:
                self.bboxes_by_image_id = json.load(f)


    def __len__(self):
        return len(self.master_df)

    def __getitem__(self, idx):
        # Load image_name and labels
        row = self.master_df.iloc[idx]

        # Extract single-class label (study-level)
        label = int(row['disease'])

        # Load image
        image_fpath = f'{row["image_fpath"]}.png'
        image = load_image(os.path.join(self.image_dir, image_fpath), self.image_format)
        image = self.transform(image)
        # shape: n_channels, height, width
        # (n_channels=3 if image_format==RGB; n_channels=1 if L or I)

        # Get bboxes
        if self.use_bboxes:
            original_size = (row['original_height'], row['original_width'])
            target_size = image.size()[-2:]

            image_id = row['image_id']
            raw_bboxes = self.bboxes_by_image_id.get(image_id, [])

            bboxes = scale_bboxes(raw_bboxes, original_size, target_size)
            # bboxes is a list of dicts with keys ("x", "y", "width", "height")
        else:
            # -1 will be collated into the dataloader without issue
            bboxes = -1

        return BatchItem(
            image=image,
            labels=label,
            image_fname=image_fpath,
            bboxes=bboxes,
        )

    def get_labels_presence_for(self, target_label):
        """Returns a list of tuples (idx, 0/1) indicating presence/absence of a
            label for each sample.
        """
        if target_label == 'No Finding':
            return self.get_presence_for_no_finding()

        if isinstance(target_label, int):
            target_label = self.labels[target_label]

        return list(self.master_df[target_label].items())

    def get_presence_for_no_finding(self):
        some_disease = self.master_df[self.labels].max(axis=1)
        # series with (index, some_disease_present)

        no_finding = 1 - some_disease

        return list(no_finding.items())
