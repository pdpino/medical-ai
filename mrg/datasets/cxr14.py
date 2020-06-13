import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
# from torchvision.transforms import functional as F
import pandas as pd
# import numpy as np
from PIL import Image
import os
import random


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

DATASET_DIR = os.environ['DATASET_DIR_CXR14']

def _get_default_image_transformation(image_size=512):
    mean = 0.50576189
    return transforms.Compose([transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize([mean], [1.])
                              ])

class CXR14Dataset(Dataset):

    def __init__(self, dataset_type='train', diseases=None, max_images=None):
        """Create a Dataset object."""
        if dataset_type not in ['train', 'val', 'test']:
            raise ValueError('No such type, must be train, val, or test')
        
        self.dataset_type = dataset_type
        self.image_format = 'RGB'
        self.transform = _get_default_image_transformation()
        
        self.image_dir = os.path.join(DATASET_DIR, 'images')

        # Load csv files
        labels_fname = os.path.join(DATASET_DIR, dataset_type + '_label.csv')
        self.label_index = pd.read_csv(labels_fname, header=0)
        
        bbox_fname = os.path.join(DATASET_DIR, 'BBox_List_2017.csv')
        self.bbox_index = pd.read_csv(bbox_fname, header=0)
        
        # Drop Bbox file unnamed columns (are empty)
        drop_unnamed = [col for col in self.bbox_index.columns if col.startswith('Unnamed')]
        self.bbox_index.drop(drop_unnamed, axis=1, inplace=True)
        
        # Choose diseases names
        if not diseases:
            self.labels = list(CXR14_DISEASES)
        else:
            # Keep only the ones that exist
            self.labels = [d for d in diseases if d in CXR14_DISEASES]
            
            not_found_diseases = list(set(self.labels) - set(CXR14_DISEASES))
            if not_found_diseases:
                print(f'Diseases not found: {not_found_diseases}(ignoring)')
            

        self.n_diseases = len(self.labels)
        
        # Filter labels DataFrame
        columns = ['FileName'] + self.labels
        self.label_index = self.label_index[columns]

        # Keep only the images in the directory # and max_images
        available_images = set(os.listdir(self.image_dir))
        labeled_images = set(self.label_index['FileName']).intersection(available_images)
        if max_images:
            labeled_images = set(list(labeled_images)[:max_images])

        self.label_index = self.label_index.loc[self.label_index['FileName'].isin(labeled_images)]
        
        # Keep bbox_index with available images
        self.bbox_index = self.bbox_index.loc[self.bbox_index['Image Index'].isin(available_images)]
        
        # Precompute items' metadata
        self.precompute_metadata()
        
    def size(self):
        n_images, _ = self.label_index.shape
        return (n_images, self.n_diseases)

    def get_by_name(self, image_name, chosen_diseases=None):
        idx = self.names_to_idx[image_name]

        image, labels, image_name, bboxes, bbox_valid = self[idx]
        
        if chosen_diseases is not None:
            labels = [
                label
                for index, label in enumerate(labels)
                if self.labels[index] in chosen_diseases
            ]
        
        return image, labels, image_name, bboxes, bbox_valid
        
    
    def __len__(self):
        n_samples, _ = self.label_index.shape
        return n_samples

    def __getitem__(self, idx):
        image_name, labels, bboxes, bbox_valid = self.precomputed[idx]
        
        image_fname = os.path.join(self.image_dir, image_name)
        try:
            image = Image.open(image_fname).convert(self.image_format)
        except OSError as e:
            print(f'({self.dataset_type}) Failed to load image, may be broken: {image_fname}')
            print(e)

            # FIXME: a way to ignore the image during training? (though it may broke other things)
            raise

        image = self.transform(image)
        
        return image, labels, image_name, bboxes, bbox_valid
    
    def precompute_metadata(self):
        self.precomputed = []
        self.names_to_idx = dict()
        for idx in range(len(self)):
            item = self.precompute_item_metadata(idx)
            image_name = item[0]

            self.precomputed.append(item)
            
            self.names_to_idx[image_name] = idx


    def precompute_item_metadata(self, idx):
        row = self.label_index.iloc[idx]
        
        # Image name
        image_name = row[0]

        # Extract labels
        labels = row[self.labels].to_numpy().astype('int')
        
        # Get bboxes
        bboxes = torch.zeros(self.n_diseases, 4) # 4: x, y, w, h
        bbox_valid = torch.zeros(self.n_diseases)

        rows = self.bbox_index.loc[self.bbox_index['Image Index']==image_name]
        for _, row in rows.iterrows():
            _, disease_name, x, y, w, h = row
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            
            # TODO: fix in the BBox csv file?
            if disease_name == 'Infiltrate':
                disease_name = 'Infiltration'

            if disease_name not in self.labels:
                continue

            disease_index = self.labels.index(disease_name)
            for j, value in enumerate([x, y, w, h]):
                bboxes[disease_index, j] = value

            bbox_valid[disease_index] = 1
        
        return image_name, labels, bboxes, bbox_valid


class CXR14UnbalancedSampler(Sampler):
    def __init__(self, cxr_dataset, max_os=None):
        total_samples = len(cxr_dataset)
        
        # Resample the indexes considering the first disease
        disease = cxr_dataset.labels[0]
        
        indexes_with_label = list(enumerate(cxr_dataset.label_index[disease]))
        
        positives = sum(label for idx, label in indexes_with_label)
        negatives = total_samples - positives
        ratio = negatives // positives if positives > 0 else 1
        
        OVERSAMPLE_LABEL = 1
        UNDERSAMPLE_LABEL = 0

        if ratio < 1:
            OVERSAMPLE_LABEL = 0
            UNDERSAMPLE_LABEL = 1
            

        # Set a maximum ratio for oversampling
        # note that it only affects ratio > 1 (i.e. oversampling positive samples)
        if max_os is not None:
            ratio = min(ratio, max_os)
        
        self.resampled_indexes = []
        
        for idx, label in indexes_with_label:
            if label == UNDERSAMPLE_LABEL:
                self.resampled_indexes.append(idx)
            elif label == OVERSAMPLE_LABEL:
                for _ in range(ratio):
                    self.resampled_indexes.append(idx)
                    
        random.shuffle(self.resampled_indexes)
        
        print(f'\tOversampling ratio: {ratio}, total {len(self.resampled_indexes)} samples (original {total_samples})')

    
    def __len__(self):
        return len(self.resampled_indexes)
    
    def __iter__(self):
        return iter(self.resampled_indexes)
        