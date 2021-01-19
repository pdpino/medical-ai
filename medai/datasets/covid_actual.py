import os
import logging
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

from medai.datasets.common import BatchItem

DATASET_DIR = os.environ.get('DATASET_DIR_COVID_ACTUAL')

LOGGER = logging.getLogger(__name__)

def _get_default_image_transformation(image_size=(512, 512)):
    # FIXME: wrong values
    mean = [0.5086, 0.5086, 0.5086]
    sd = [0.0416, 0.0416, 0.0416]
    return transforms.Compose([transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean, sd)
                              ])

class CovidActualDataset(Dataset):
    def __init__(self, dataset_type='train', image_size=(512, 512), **unused_kwargs):
        super().__init__()

        raise NotImplementedError('CovidActualDataset mean and std')

        # pylint: disable=unreachable
        if DATASET_DIR is None:
            raise Exception('DATASET_DIR_COVID_ACTUAL not found in env variables')

        if dataset_type not in ['train', 'val', 'test']:
            raise ValueError('No such type, must be train, val, or test')

        # HACK
        dataset_type = 'test'
        LOGGER.info('\tCovid-actual only has test dataset')

        if unused_kwargs.get('labels', None) is not None:
            LOGGER.warning('Ignoring labels in CovidActualDataset, not implemented yet')

        self.dataset_type = dataset_type
        self.image_format = 'RGB'
        self.image_size = image_size
        self.transform = _get_default_image_transformation(self.image_size)

        # FIXME: map these names to regular names
        self.labels = ['COVID-19', 'pneumonia', 'No finding']
        self.multilabel = False

        # Load metadata
        meta_path = os.path.join(DATASET_DIR, 'metadata.csv')
        df = pd.read_csv(meta_path)
        df = df[['finding', 'imagename']]
        self.df = df.loc[(df['finding'] == 'COVID-19') | (df['finding'] == 'No finding')]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row['imagename']
        finding = row['finding']
        if finding == 'COVID-19':
            label = 0
        else:
            label = 2

        image_path = os.path.join(DATASET_DIR, 'images', image_name)
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
