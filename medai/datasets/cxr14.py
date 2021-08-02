import os
import json
import logging
import torch
from torch.utils.data import Dataset
import pandas as pd

from medai.datasets.common import (
    BatchItem,
    CXR14_DISEASES,
    JSRT_ORGANS,
    UP_TO_DATE_MASKS_VERSION,
)
from medai.utils.images import (
    get_default_image_transform,
    load_image,
    get_default_mask_transform,
)

LOGGER = logging.getLogger(__name__)

DATASET_DIR = os.environ.get('DATASET_DIR_CXR14')

_BROKEN_IMAGES = set((
    # Looks empty (all gray)
    '00007160_002.png',
    '00012249_001.png',
    # Lateral images # TODO: mark as lateral!
    # NOTE: this is not a comprehensive list of lateral images,
    # but images that have been casually found.
    # They are poorly labeled in the Data_Entry_2017.csv file
    # (i.e. are marked as frontal)
    '00013774_015.png',
    # Color-inverted:
    '00020006_001.png',
    '00021201_042.png',
    # One lung patient:
    '00029041_016.png',
    '00029041_017.png',
))

_DATASET_MEAN = 0.5058
_DATASET_STD = 0.232

_ORIGINAL_IMAGE_SIZE = 1024
# FIXME: this is not the correct image size for all images!!

def _calculate_bbox_scale(image_size):
    height, width = image_size
    if height == width:
        return _ORIGINAL_IMAGE_SIZE // height

    scale_height = _ORIGINAL_IMAGE_SIZE / height
    scale_width = _ORIGINAL_IMAGE_SIZE / width

    return torch.tensor((scale_height, scale_width, scale_height, scale_width)) # pylint: disable=not-callable


class CXR14Dataset(Dataset):
    organs = list(JSRT_ORGANS)
    dataset_dir = DATASET_DIR

    def __init__(self, dataset_type='train', labels=None, max_samples=None,
                 image_size=(512, 512), norm_by_sample=False, image_format='RGB',
                 masks=False, images_version=None, masks_version=UP_TO_DATE_MASKS_VERSION,
                 xrv_norm=False, seg_multilabel=True, bboxes=False,
                 crop_center=None,
                 **unused_kwargs):
        super().__init__()

        if DATASET_DIR is None:
            raise Exception('DATASET_DIR_CXR14 not found in env variables')

        self.dataset_type = dataset_type
        self.image_format = image_format
        self.image_size = image_size if crop_center is None else (crop_center, crop_center)
        self.transform = get_default_image_transform(
            self.image_size,
            norm_by_sample=norm_by_sample,
            crop_center=crop_center,
            mean=_DATASET_MEAN,
            std=_DATASET_STD,
            xrv_norm=xrv_norm,
        )

        self.image_dir = os.path.join(
            DATASET_DIR,
            'images'
            if images_version is None else f'images-{images_version}',
        )
        if not os.path.isdir(self.image_dir):
            raise ValueError(
                f'Image version not found: {images_version} (in {self.image_dir})',
            )

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

            not_found_diseases = list(set(labels) - set(CXR14_DISEASES))
            if not_found_diseases:
                LOGGER.warning('Diseases not found: %s (ignoring)', not_found_diseases)

        self.n_diseases = len(self.labels)
        self.multilabel = True # CL_multilabel
        self.seg_multilabel = seg_multilabel

        assert self.n_diseases > 0, 'No diseases selected!'

        # Filter labels DataFrame
        columns = ['FileName'] + self.labels
        self.label_index = self.label_index[columns]

        # Remove broken images
        split_images = set(split_images) - _BROKEN_IMAGES

        # Keep only the images in the directory
        available_images = split_images.intersection(set(os.listdir(self.image_dir)))
        available_images = set(self.label_index['FileName']).intersection(available_images)

        if len(split_images) > len(available_images):
            missing_images = set(split_images) - available_images
            LOGGER.warning('Warning: %s images are not available:\n%s',
                            len(missing_images), missing_images)

        # Keep only max_samples images
        if max_samples:
            available_images = set(list(available_images)[:max_samples])

        self.label_index = self.label_index.loc[self.label_index['FileName'].isin(available_images)]

        self.label_index.reset_index(drop=True, inplace=True)

        self.enable_masks = masks
        if self.enable_masks:
            self.transform_mask = get_default_mask_transform(
                image_size,
                self.seg_multilabel,
                len(self.organs),
                crop_center=crop_center,
            )

            self.masks_dir = os.path.join(DATASET_DIR, 'masks', masks_version)

            assert os.path.isdir(self.masks_dir), f'Masks {self.masks_dir} do not exist'


        self.enable_bboxes = bboxes
        self.bbox_scale = _calculate_bbox_scale(self.image_size)

    def __len__(self):
        n_samples, _ = self.label_index.shape
        return n_samples

    def __getitem__(self, idx):
        # Load image_name and labels
        row = self.label_index.iloc[idx]

        # Image name
        image_fname = row[0]

        # Extract labels
        # pylint: disable=not-callable
        labels = torch.tensor(row[self.labels], dtype=torch.uint8)

        image_fpath = os.path.join(self.image_dir, image_fname)
        image = load_image(image_fpath, self.image_format)

        image = self.transform(image)

        masks = self.load_mask(image_fname) if self.enable_masks else -1

        bboxes, bboxes_valid = self.get_bboxes(image_fname)

        return BatchItem(
            image=image,
            labels=labels,
            masks=masks,
            bboxes=bboxes,
            bboxes_valid=bboxes_valid,
            image_fname=image_fname,
        )

    def get_item_by_image_name(self, image_name):
        if not image_name.endswith('.png'):
            image_name = f'{image_name}.png'
        rows = self.label_index.loc[self.label_index['FileName'] == image_name]
        assert len(rows) == 1, f'Found not exactly 1: {rows}'
        idx = rows.index[0]
        return self[idx]

    def get_bboxes(self, image_name):
        # REVIEW: precompute this?
        if not self.enable_bboxes:
            return -1, -1

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

        mask = load_image(filepath, 'L')
        mask = self.transform_mask(mask)
        return mask

    def get_labels_presence_for(self, target_label):
        """Returns a list of tuples (idx, 0/1) indicating presence/absence of a
            label for each sample.
        """
        if target_label == 'No Finding':
            return self.get_presence_for_no_finding()

        if isinstance(target_label, int):
            target_label = self.labels[target_label]

        return list(self.label_index[target_label].items())

    def get_presence_for_no_finding(self):
        some_disease = self.label_index[self.labels].max(axis=1)
        # series with (index, some_disease_present)

        no_finding = 1 - some_disease

        return list(no_finding.items())

    def image_names_to_indexes(self, image_names):
        if isinstance(image_names, str):
            image_names = (image_names,)
        image_names = set(image_names)

        rows = self.label_index.loc[self.label_index['FileName'].isin(image_names)]
        return rows.index
