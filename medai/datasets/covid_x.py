from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import pandas as pd


DATASET_DIR = os.environ.get('DATASET_DIR_COVID_X')


def _get_default_image_transformation(image_size=(512, 512)):
    mean = [0.4919, 0.4920, 0.4920]
    sd = [0.0467, 0.0467, 0.0467]
    return transforms.Compose([transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean, sd)
                              ])

class CovidXDataset(Dataset):
    def __init__(self, dataset_type='train', max_samples=None, image_size=(512, 512), **kwargs):
        if DATASET_DIR is None:
            raise Exception(f'DATASET_DIR_COVID_X not found in env variables')

        if dataset_type not in ['train', 'val', 'test']:
            raise ValueError('No such type, must be train, val or test')

        if kwargs.get('labels', None) is not None:
            print('Labels selection in CovidX dataset is not implemented yet, ignoring')

        self.dataset_type = dataset_type
        self.image_format = 'RGB'
        self.image_size = image_size
        self.transform = _get_default_image_transformation(self.image_size)

        self.multilabel = False

        # If train or val, load from train
        dataset_type_load = dataset_type
        if dataset_type == 'val':
            dataset_type_load = 'train'

        # Images folder
        self.images_dir = os.path.join(DATASET_DIR, dataset_type_load)

        # Load metadata
        labels_fpath = os.path.join(DATASET_DIR, f'{dataset_type_load}_split.txt')
        columns = ['patient_id', 'image_name', 'label', 'source']
        self._metadata_df = pd.read_csv(labels_fpath, sep=' ', header=None, names=columns)

        # Save labels
        self._metadata_df.replace('COVID-19', 'covid', inplace=True)
        self.labels = ['covid', 'pneumonia', 'normal']
        self._label_to_idx = {label: idx for idx, label in enumerate(self.labels)}

        # Assert correct labels
        df_labels = set(self._metadata_df['label'])
        correct_labels = set(self.labels)
        if df_labels != correct_labels:
            raise Exception(f'Labels from dataset are not correct: {df_labels}')

        # Split train and val
        if dataset_type in ['train', 'val']:
            with open(os.path.join(DATASET_DIR, f'further_{dataset_type}_split.txt')) as f:
                selected_images = [l.strip() for l in f.readlines()]
        else:
            # if test, keep all images
            selected_images = list(self._metadata_df['image_name'])

        # Keep only max images
        if max_samples is not None:
            selected_images = selected_images[:max_samples]

        # Actually filter images
        selected_images = set(selected_images)
        self._metadata_df = self._metadata_df.loc[
            self._metadata_df['image_name'].isin(selected_images)
        ]

        self._metadata_df.reset_index(inplace=True)

        
    def size(self):
        return len(self._metadata_df), len(self.labels)

    def __len__(self):
        return len(self._metadata_df)

    def __getitem__(self, idx):
        row = self._metadata_df.iloc[idx]
        image_name = row['image_name'] # includes extension
        label_str = row['label']
        label = self._label_to_idx[label_str]

        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert(self.image_format)
        image = self.transform(image)

        return image, label

    def get_labels_presence_for(self, target_label):
        """Returns a list of tuples (idx, 0/1) indicating presence/absence of a
            label for each sample.
        """
        if isinstance(target_label, int):
            target_label = self.labels[target_label]

        return [
            (idx, int(target_label == row['label']))
            for idx, row in self._metadata_df.iterrows()
        ]
