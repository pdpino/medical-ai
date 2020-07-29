import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
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
                 image_size=(512, 512)):
        if DATASET_DIR is None:
            raise Exception(f'DATASET_DIR_CXR14 not found in env variables')

        if dataset_type not in ['train', 'val', 'test']:
            raise ValueError('No such type, must be train, val, or test')
        
        self.dataset_type = dataset_type
        self.image_format = 'RGB'
        self.image_size = image_size
        self.transform = _get_default_image_transformation(self.image_size)
        
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
        available_images = set(os.listdir(self.image_dir))
        available_images = set(self.label_index['FileName']).intersection(available_images)

        # Keep only max_samples images
        if max_samples:
            available_images = set(list(available_images)[:max_samples])

        self.label_index = self.label_index.loc[self.label_index['FileName'].isin(available_images)]
        
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

        return image, labels, bboxes, bbox_valid, image_name
    
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

        return image_name, labels, bboxes, bbox_valid # DEBUG: loads faster

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

    def get_labels_presence_for(self, target_label):
        """Returns a list of tuples (idx, 0/1) indicating presence/absence of a
            label for each sample.
        """
        if isinstance(target_label, int):
            target_label = self.labels[target_label]

        return list(enumerate(self.label_index[target_label]))