import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from ignite.utils import to_onehot
import pandas as pd
from PIL import Image

from medai.datasets.common import (
    BatchItem,
    JSRT_ORGANS,
    ORGAN_BACKGROUND,
    ORGAN_HEART,
    ORGAN_LEFT_LUNG,
    ORGAN_RIGHT_LUNG,
)
from medai.utils.images import get_default_image_transform

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

_DISEASE_TO_ORGAN = {
    disease: (ORGAN_RIGHT_LUNG, ORGAN_LEFT_LUNG)
    for disease in CXR14_DISEASES
}
_DISEASE_TO_ORGAN['Cardiomegaly'] = (ORGAN_HEART,)
_DISEASE_TO_ORGAN['Hernia'] = (ORGAN_BACKGROUND, ORGAN_HEART, ORGAN_RIGHT_LUNG, ORGAN_LEFT_LUNG)


DATASET_DIR = os.environ.get('DATASET_DIR_CXR14')

_DATASET_MEAN = 0.5058
_DATASET_STD = 0.232

class CXR14Dataset(Dataset):
    organs = list(JSRT_ORGANS)
    _disease_to_organs = _DISEASE_TO_ORGAN

    def __init__(self, dataset_type='train', labels=None, max_samples=None,
                 image_size=(512, 512), norm_by_sample=False, image_format='RGB',
                 masks=False,
                 **unused_kwargs):
        super().__init__()

        if DATASET_DIR is None:
            raise Exception('DATASET_DIR_CXR14 not found in env variables')

        self.dataset_type = dataset_type
        self.image_format = image_format
        self.image_size = image_size
        self.transform = get_default_image_transform(
            self.image_size,
            norm_by_sample=norm_by_sample,
            mean=_DATASET_MEAN,
            std=_DATASET_STD,
        )

        self.image_dir = os.path.join(DATASET_DIR, 'images')
        self.masks_dir = os.path.join(DATASET_DIR, 'masks')

        # Load split images
        SPLITS_DIR = os.path.join(DATASET_DIR, 'splits')
        split_fpath = os.path.join(SPLITS_DIR, f'{dataset_type}.txt')
        if not os.path.isfile(split_fpath):
            _AVAILABLE_SPLITS = [
                split.replace('.txt', '')
                for split in os.listdir(SPLITS_DIR)
            ]
            raise ValueError(f'No such type, must be one of {_AVAILABLE_SPLITS}')

        with open(split_fpath, 'r') as f:
            split_images = [l.strip() for l in f.readlines()]


        # Load csv files
        labels_fname = os.path.join(DATASET_DIR, 'label_index.csv')
        self.label_index = pd.read_csv(labels_fname, header=0)

        # Load Bbox JSON
        bboxes_fpath = os.path.join(DATASET_DIR, 'bbox_by_image_by_disease.json')
        with open(bboxes_fpath, 'r') as f:
            self.bboxes_by_image = json.load(f)

        # Choose diseases names
        if not labels:
            self.labels = list(CXR14_DISEASES)
        else:
            # Keep only the ones that exist
            self.labels = [d for d in labels if d in CXR14_DISEASES]

            not_found_diseases = list(set(self.labels) - set(CXR14_DISEASES))
            if not_found_diseases:
                print(f'Diseases not found: {not_found_diseases}(ignoring)')

        self.n_diseases = len(self.labels)
        self.multilabel = True

        assert self.n_diseases > 0, 'No diseases selected!'

        # Filter labels DataFrame
        columns = ['FileName'] + self.labels
        self.label_index = self.label_index[columns]

        # Keep only the images in the directory
        available_images = set(split_images).intersection(set(os.listdir(self.image_dir)))
        available_images = set(self.label_index['FileName']).intersection(available_images)

        # Keep only max_samples images
        if max_samples:
            available_images = set(list(available_images)[:max_samples])

        self.label_index = self.label_index.loc[self.label_index['FileName'].isin(available_images)]

        self.label_index.reset_index(drop=True, inplace=True)

        self.enable_masks = masks
        if self.enable_masks:
            self.transform_mask = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ])

    def __len__(self):
        n_samples, _ = self.label_index.shape
        return n_samples

    def __getitem__(self, idx):
        # Load image_name and labels
        row = self.label_index.iloc[idx]

        # Image name
        image_name = row[0]

        # Extract labels
        labels = row[self.labels].to_numpy().astype('int')

        # Load image
        image_fname = os.path.join(self.image_dir, image_name)
        try:
            image = Image.open(image_fname).convert(self.image_format)
        except OSError as e:
            print(f'({self.dataset_type}) Failed to load image, may be broken: {image_fname}')
            print(e)

            # FIXME: a way to ignore the image during training? (though it may break other things)
            raise

        image = self.transform(image)

        masks = self.load_mask(image_name) if self.enable_masks else -1

        # Load bboxes # REVIEW: precompute this?
        raw_bboxes = self.bboxes_by_image.get(image_name, {})
        bboxes = []
        bboxes_valid = []

        for label in self.labels:
            bbox = raw_bboxes.get(label, None)
            if bbox is None:
                bboxes_valid.append(0)
                bboxes.append([0, 0, 0, 0])
            else:
                bboxes_valid.append(1)
                bboxes.append(bbox)

        # pylint: disable=not-callable
        bboxes_valid = torch.tensor(bboxes_valid).float()
        bboxes = torch.tensor(bboxes).float()

        return BatchItem(
            image=image,
            labels=labels,
            masks=masks,
            bboxes=bboxes,
            bboxes_valid=bboxes_valid,
            image_fname=image_name,
        )

    def load_mask(self, image_name):
        if not self.enable_masks:
            return None

        filepath = os.path.join(self.masks_dir, image_name)

        if not os.path.isfile(filepath):
            print('No such file: ', filepath)
            return None

        mask = Image.open(filepath).convert('L')
        mask = self.transform_mask(mask)
        # shape: n_channels=1, height, width

        mask = (mask * 255).long()
        # shape: 1, height, width

        mask = to_onehot(mask, len(self.organs))
        # shape: 1, n_organs, height, width

        mask = mask.squeeze(0)
        # shape: n_organs, height, width

        return mask

    def get_labels_presence_for(self, target_label):
        """Returns a list of tuples (idx, 0/1) indicating presence/absence of a
            label for each sample.
        """
        if isinstance(target_label, int):
            target_label = self.labels[target_label]

        return list(enumerate(self.label_index[target_label]))

    def reduce_masks_for_disease(self, label, sample_masks):
        """Reduce a tensor of organ masks for a given disease

        Args:
            label -- disease (str)
            sample_masks -- tensor of shape (*, n_organs, height, width)
                Notice it may be masks for a batch (i.e. batch_size at front), or for one sample
        """
        # Get organ idxs
        organs_idxs = torch.tensor([ # pylint: disable=not-callable
            self.organs.index(organ_name)
            for organ_name in self._disease_to_organs[label]
        ]).to(sample_masks.device)

        # Select organs
        mask = sample_masks.index_select(dim=-3, index=organs_idxs)
        # shape: *, n_selected_organs, height, width

        # Add-up (assume sum wont be more than 1)
        mask = mask.sum(dim=-3)
        # shape: *, height, width

        return mask
