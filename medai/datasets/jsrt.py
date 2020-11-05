import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from medai.datasets.common import BatchItem
from medai.utils.images import get_default_image_transform

DATASET_DIR = os.environ.get('DATASET_DIR_JSRT', None)

_DATASET_MEAN = 0.5937
_DATASET_STD = 0.2729

ORGANS = [
    'heart',
    'right lung',
    'left lung',
    ## Not used for now:
    # 'right clavicle',
    # 'left clavicle',
]

class JSRTDataset(Dataset):
    def __init__(self, dataset_type='all', image_size=(512, 512), norm_by_sample=False, **unused):

        assert DATASET_DIR is not None, 'DATASET_DIR is None'

        self.images_dir = os.path.join(DATASET_DIR, 'images')

        metadata_fpath = os.path.join(DATASET_DIR, 'jsrt_metadata.csv')

        self.metadata = pd.read_csv(metadata_fpath)

        self.images_names = list(self.metadata['study_id'])

        self.dataset_type = dataset_type
        if dataset_type != 'all':
            split_fpath = os.path.join(DATASET_DIR, 'splits', f'{dataset_type}.txt')
            with open(split_fpath, 'r') as f:
                selected_images = [l.strip() for l in f.readlines()]

            self.images_names = [
                name
                for name in self.images_names
                if name in selected_images
            ]

        self.image_size = image_size
        self.transform = get_default_image_transform(
            self.image_size,
            norm_by_sample=norm_by_sample,
            mean=_DATASET_MEAN,
            std=_DATASET_STD,
        )
        self.transform_mask = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

        self.seg_labels = ['background'] + list(ORGANS)


    def __len__(self):
        return len(self.images_names)


    def __getitem__(self, idx):
        image_name = self.images_names[idx]

        image_fpath = os.path.join(self.images_dir, image_name)
        image = Image.open(image_fpath)

        image = self.transform(image)

        masks = self.get_masks(image_name)

        return BatchItem(
            image=image,
            filename=image_name,
            masks=masks,
        )


    def get_masks(self, image_name):
        """Load segmentation masks for an image."""
        image_name_wo_ext = image_name[:-4]

        is_even = int(image_name_wo_ext[-1]) % 2 == 0
        fold = 'fold2' if is_even else 'fold1'

        image_name_gif = f'{image_name_wo_ext}.gif'
        masks_folder = os.path.join(DATASET_DIR, 'scratch', fold, 'masks')

        overall_mask = torch.zeros(*self.image_size).long()

        for index, organ in enumerate(self.seg_labels):
            if organ == 'background':
                continue

            fpath = os.path.join(masks_folder, organ, image_name_gif)

            mask = Image.open(fpath)
            mask = self.transform_mask(mask) # shape: 1, height, width

            mask = mask.squeeze(0) # shape: height, width

            overall_mask[mask > 0] = index

        return overall_mask
