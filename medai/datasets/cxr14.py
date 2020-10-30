import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
import json
import random

from medai.datasets.common import BatchItem

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

DATASET_DIR = os.environ.get('DATASET_DIR_CXR14')

def _get_default_image_transformation(image_size=(512, 512)):
    mean = 0.4980 # 0.50576189
    sd = 0.0458
    return transforms.Compose([transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize([mean], [sd]),
                              ])

class CXR14Dataset(Dataset):
    def __init__(self, dataset_type='train', labels=None, max_samples=None,
                 image_size=(512, 512), **unused):
        if DATASET_DIR is None:
            raise Exception(f'DATASET_DIR_CXR14 not found in env variables')

        self.dataset_type = dataset_type
        self.image_format = 'RGB'
        self.image_size = image_size
        self.transform = _get_default_image_transformation(self.image_size)
        
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

        bboxes_valid = torch.tensor(bboxes_valid).float()
        bboxes = torch.tensor(bboxes).float()

        return BatchItem(
            image=image,
            labels=labels,
            bboxes=bboxes,
            bboxes_valid=bboxes_valid,
            filename=image_name,
        )

    def get_labels_presence_for(self, target_label):
        """Returns a list of tuples (idx, 0/1) indicating presence/absence of a
            label for each sample.
        """
        if isinstance(target_label, int):
            target_label = self.labels[target_label]

        return list(enumerate(self.label_index[target_label]))