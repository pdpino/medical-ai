from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd

from medai.datasets.common import BatchItem
from medai.utils.images import get_default_image_transform, load_image

DATASET_DIR = os.environ.get('DATASET_DIR_COVID_UC')

_DATASET_MEAN_STD = {
    'frontal': (0.4231, 0.1256),
    'all': (0.3666, 0.1295),
}


class CovidUCDataset(Dataset):
    def __init__(self, dataset_type='train', max_samples=None,
                 image_size=(512, 512), frontal_only=False, image_format='RGB',
                 norm_by_sample=False, **unused):
        if DATASET_DIR is None:
            raise Exception(f'DATASET_DIR_COVID_UC not found in env variables')

        _availables = ['all', 'train', 'val', 'test']
        if dataset_type not in _availables:
            raise ValueError(f'No such type "{dataset_type}", must be in {_availables}')

        self.dataset_type = dataset_type
        self.image_format = image_format
        self.image_size = image_size

        mean, std = _DATASET_MEAN_STD['frontal' if frontal_only else 'all']
        self.transform = get_default_image_transform(
            self.image_size,
            norm_by_sample=norm_by_sample,
            mean=mean,
            std=std,
        )

        self.multilabel = False

        # Images folder
        self.images_dir = os.path.join(DATASET_DIR, 'images')

        # Load metadata
        labels_fpath = os.path.join(DATASET_DIR, 'metadata.csv')
        self._metadata_df = pd.read_csv(labels_fpath, index_col=0)

        # Keep only frontal
        if frontal_only:
            self._metadata_df = self._metadata_df.loc[self._metadata_df['view'].str.contains('P')]

        selected_images = list(self._metadata_df['image_name'])

        # Load split
        if dataset_type != 'all':
            split_fpath = os.path.join(DATASET_DIR, f'{dataset_type}.txt')
            with open(split_fpath, 'r') as f:
                split_images = set([l.strip() for l in f.readlines()])

            selected_images = [i for i in selected_images if i in split_images]


        # Labels
        # TODO: check labels used (Non-COVID === pneumonia, for now)
        self.labels = ['covid', 'Non-COVID', 'normal']
        self._label_to_idx = {label: idx for idx, label in enumerate(self.labels)}

        # Keep only max images
        if max_samples is not None:
            selected_images = selected_images[:max_samples]

        # Actually filter images
        selected_images = set(selected_images)
        self._metadata_df = self._metadata_df.loc[
            self._metadata_df['image_name'].isin(selected_images)
        ]

        self._metadata_df.reset_index(inplace=True, drop=True)


    def __len__(self):
        return len(self._metadata_df)

    def __getitem__(self, idx):
        row = self._metadata_df.iloc[idx]
        image_name = row['image_name'] # includes extension
        label_str = row['Resultado consenso BSTI']
        label = self._label_to_idx[label_str]

        image_path = os.path.join(self.images_dir, image_name)
        image = load_image(image_path, self.image_format)
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
            (idx, int(target_label == row['Resultado consenso BSTI']))
            for idx, row in self._metadata_df.iterrows()
        ]
