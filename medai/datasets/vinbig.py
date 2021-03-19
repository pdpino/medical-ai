import os
import json
import logging
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image

from medai.datasets.common import (
    BatchItem,
    VINBIG_DISEASES,
)
from medai.utils.images import get_default_image_transform

LOGGER = logging.getLogger(__name__)

DATASET_DIR = os.environ.get('DATASET_DIR_VINBIG')

_DATASET_MEAN = 0.5489
_DATASET_STD = 0.2498

class VinBigDataset(Dataset):
    dataset_dir = DATASET_DIR

    def __init__(self, dataset_type='train', max_samples=None,
                 image_size=(512, 512), norm_by_sample=False, image_format='RGB',
                 masks=False, bboxes=False, **unused_kwargs):
        super().__init__()

        if DATASET_DIR is None:
            raise Exception('DATASET_DIR_VINBIG not found in env variables')

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
            split_images = [l.strip().replace('.png', '') for l in f.readlines()]

        if dataset_type == 'test':
            LOGGER.warning('Vinbig test-dataset labels are dummy (not real)!!')

        # Load csv files
        fpath = os.path.join(DATASET_DIR, 'labels.csv')
        self.label_index = pd.read_csv(fpath, header=0)

        # Load Bbox JSON
        fpath = os.path.join(DATASET_DIR, 'bboxes.json')
        with open(fpath, 'r') as f:
            self.bboxes_by_image = json.load(f)

        # Choose diseases names
        self.labels = list(VINBIG_DISEASES[:-1])
        self.multilabel = True

        # Keep only max_samples images
        available_images = split_images
        if max_samples:
            available_images = available_images[:max_samples]

        self.label_index = self.label_index.loc[self.label_index['image_id'].isin(available_images)]

        self.label_index.reset_index(drop=True, inplace=True)

        self.enable_bboxes = bboxes
        self.enable_masks = masks
        if self.enable_masks:
            self.transform_mask = transforms.Resize(image_size)

        # Used for evaluation with mAP-coco
        self.coco_gt_df = self._load_gt_df_for_coco(self.label_index['image_id'])


    def __len__(self):
        return len(self.label_index)

    def __getitem__(self, idx):
        # Load image_name and labels
        row = self.label_index.iloc[idx]

        # Image name
        image_id = row['image_id']

        # Extract labels
        labels = row[self.labels].to_numpy().astype('int')

        # Load image
        image_fpath = os.path.join(self.image_dir, f'{image_id}.png')
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

        original_size = tuple(image.size)
        image = self.transform(image)

        masks = self.load_mask(image_id, original_size) if self.enable_masks else -1

        bboxes = self.load_scaled_bboxes(image_id, original_size) if self.enable_bboxes else -1

        return BatchItem(
            image=image,
            labels=labels,
            masks=masks,
            image_fname=image_id,
            bboxes=bboxes,
        )

    def load_mask(self, image_id, original_size):
        bboxes = self.bboxes_by_image.get(image_id, [])
        diseases_with_bb = set()
        mask = torch.ones(len(self.labels), *original_size)

        for bbox in bboxes:
            disease_id = bbox[0]
            x_min, y_min, x_max, y_max = bbox[1:]

            if disease_id not in diseases_with_bb:
                mask[disease_id] = 0
            diseases_with_bb.add(disease_id)

            mask[disease_id, y_min:y_max, x_min:x_max] = 1

        mask = self.transform_mask(mask)
        # shape: n_diseases, height, width

        return mask

    def load_scaled_bboxes(self, image_id, original_size):
        bboxes = self.bboxes_by_image.get(image_id, [])

        original_height, original_width = original_size
        height, width = self.image_size

        horizontal_scale = height / original_height
        vertical_scale = width / original_width

        scaled_bboxes = []
        for bbox in bboxes:
            disease_id = bbox[0]
            x_min, y_min, x_max, y_max = bbox[1:]


            x_min = int(x_min * horizontal_scale)
            x_max = int(x_max * horizontal_scale)
            y_min = int(y_min * vertical_scale)
            y_max = int(y_max * vertical_scale)

            scaled_bboxes.append((disease_id, x_min, y_min, x_max, y_max))

        return scaled_bboxes

    def get_labels_presence_for(self, target_label):
        """Returns a list of tuples (idx, 0/1) indicating presence/absence of a
            label for each sample.
        """
        if isinstance(target_label, str) and target_label.lower() == 'no finding':
            return self.get_presence_for_no_finding()

        if isinstance(target_label, int):
            target_label = self.labels[target_label]

        return list(self.label_index[target_label].items())

    def get_presence_for_no_finding(self):
        some_disease = self.label_index[self.labels].max(axis=1)
        # series with (index, some_disease_present)

        no_finding = 1 - some_disease

        return list(no_finding.items())

    def _load_gt_df_for_coco(self, image_names):
        fpath = os.path.join(DATASET_DIR, 'true_df.csv')
        df = pd.read_csv(fpath)
        df = df.loc[df['image_id'].isin(image_names)]

        return df
