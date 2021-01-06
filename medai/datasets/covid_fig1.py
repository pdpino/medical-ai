import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

from medai.datasets.common import BatchItem

DATASET_DIR = os.environ.get('DATASET_DIR_COVID_FIG1')

def _get_default_image_transformation(image_size=(512, 512)):
    # FIXME: wrong values
    mean = [0.5309, 0.5309, 0.5309]
    sd = [0.0235, 0.0235, 0.0235]
    return transforms.Compose([transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean, sd),
                              ])

class CovidFig1Dataset(Dataset):
    def __init__(self, dataset_type='train', image_size=(512, 512), **kwargs):
        raise NotImplementedError('CovidFig1Dataset mean and std')

        if DATASET_DIR is None:
            raise Exception(f'DATASET_DIR_COVID_FIG1 not found in env variables')

        if dataset_type not in ['train', 'val', 'test']:
            raise ValueError('No such type, must be train, val, or test')

        dataset_type = 'test'
        print('\tCovid-fig1 only has test dataset')

        if kwargs.get('labels', None) is not None:
            print('Labels selection in CovidKaggle dataset is not implemented yet, ignoring')

        self.dataset_type = dataset_type
        self.image_format = 'RGB'
        self.image_size = image_size
        self.transform = _get_default_image_transformation(self.image_size)

        self.labels = ['covid', 'pneumonia', 'normal']
        self.multilabel = False

        self.images_dir = os.path.join(DATASET_DIR, 'images')
        self._available_images = os.listdir(self.images_dir)

        # Load metadata
        meta_path = os.path.join(DATASET_DIR, 'metadata.csv')
        df = pd.read_csv(meta_path, encoding='latin')
        df = df[['finding', 'patientid']]
        df.dropna(axis=0, how='any', inplace=True)

        df.replace('COVID-19', 'covid', inplace=True)
        df.replace('Pneumonia', 'pneumonia', inplace=True)
        df.replace('No finding', 'normal', inplace=True)

        df.reset_index(inplace=True, drop=True)

        self.df = df


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row['patientid']
        label = self.labels.index(row['finding'])

        if image_name + '.png' in self._available_images:
            image_name += '.png'
        elif image_name + '.jpg' in self._available_images:
            image_name += '.jpg'

        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert(self.image_format)
        image = self.transform(image)

        return BatchItem(
            image=image,
            labels=label,
            image_fname=image_name,
        )

    def get_labels_presence_for(self, target_label):
        """Returns a list of tuples (idx, 0/1) indicating presence/absence of a
            label for each sample.
        """
        if isinstance(target_label, int):
            target_label = self.labels[target_label]

        return [
            (idx, int(row['finding'] == target_label))
            for idx, row in self.df.iterrows()
        ]
