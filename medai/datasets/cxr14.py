import os
import json
import torch
import logging
from torch.utils.data import Dataset
from torchvision import transforms
from ignite.utils import to_onehot
import pandas as pd
from PIL import Image

from medai.datasets.common import (
    BatchItem,
    CXR14_DISEASES,
    JSRT_ORGANS,
)
from medai.utils.images import get_default_image_transform

DATASET_DIR = os.environ.get('DATASET_DIR_CXR14')

_DATASET_MEAN = 0.5058
_DATASET_STD = 0.232

_ORIGINAL_IMAGE_SIZE = 1024

def _calculate_bbox_scale(image_size):
    height, width = image_size
    if height == width:
        return _ORIGINAL_IMAGE_SIZE // height

    scale_height = _ORIGINAL_IMAGE_SIZE / height
    scale_width = _ORIGINAL_IMAGE_SIZE / width

    return torch.tensor((scale_height, scale_width, scale_height, scale_width)) # pylint: disable=not-callable


class CXR14Dataset(Dataset):
    organs = list(JSRT_ORGANS)

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

        if len(split_images) > len(available_images):
            missing_images = set(split_images) - available_images
            logging.warning('Warning: %s images are not available:\n%s',
                            len(missing_images), missing_images)

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

        self.bbox_scale = _calculate_bbox_scale(self.image_size)


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

        bboxes, bboxes_valid = self.get_bboxes(image_name)

        return BatchItem(
            image=image,
            labels=labels,
            masks=masks,
            bboxes=bboxes,
            bboxes_valid=bboxes_valid,
            image_fname=image_name,
        )

    def get_bboxes(self, image_name):
        # REVIEW: precompute this?

        # Load raw bboxes
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
        bboxes_valid = torch.tensor(bboxes_valid).float() # shape: n_labels
        bboxes = (torch.tensor(bboxes) / self.bbox_scale).float() # shape: n_labels, 4

        return bboxes, bboxes_valid

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
