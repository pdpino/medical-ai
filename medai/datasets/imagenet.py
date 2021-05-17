import os
import json
import pandas as pd
from torch.utils.data import Dataset

from medai.datasets.common import BatchItem
from medai.utils.images import get_default_image_transform, load_image

_AVAILABLE_SPLITS = ('val', 'train')

DATASET_DIR = os.environ.get('DATASET_DIR_IMAGENET')

_DATASET_MEAN = [0.4811, 0.4575, 0.4078]
_DATASET_STD = [0.2335, 0.2294, 0.2302]


def _keep_max_samples_balanced(df, labels, max_samples):
    """Ensure there are samples from each class.

    Prevents metrics from failing with no-positive-samples errors.
    - Precision, Recall with `is_multilabel=False` do not present errors
    - roc_auc does present errors (if there are no samples from one class)
    """
    n_samples_by_class = max_samples // len(labels)
    if n_samples_by_class < 1:
        raise Exception(f'max_samples == {max_samples} is too low, choose >1000')

    grouped_by_class = df.groupby('wnid')['image_name'].apply(list)
    grouped_by_class = grouped_by_class.to_dict()

    chosen = []
    for label_name in labels:
        # for label_name, samples_by_class in grouped_by_class.to_dict().items():
        samples_by_class = grouped_by_class[label_name]
        if len(samples_by_class) == 0:
            raise Exception(f'using max_samples, there are no samples for {label_name}')
        chosen.extend(samples_by_class[:n_samples_by_class])
    chosen = set(chosen)
    df = df.loc[df['image_name'].isin(chosen)]

    return df


class ImageNetDataset(Dataset):
    dataset_dir = DATASET_DIR
    multilabel = False
    enable_masks = False

    def __init__(self, dataset_type='train', norm_by_sample=False,
                 image_format='RGB', image_size=(256, 256), max_samples=None,
                 mini=None,
                 **unused_kwargs):
        super().__init__()

        if DATASET_DIR is None:
            raise Exception('DATASET_DIR_IMAGENET not found in env variables')

        if dataset_type not in _AVAILABLE_SPLITS:
            raise Exception(f'Split {dataset_type} not available, use one of {_AVAILABLE_SPLITS}')

        self.dataset_type = dataset_type
        self.image_format = image_format
        self.image_size = image_size
        self.transform = get_default_image_transform(
            self.image_size,
            norm_by_sample=norm_by_sample,
            mean=_DATASET_MEAN,
            std=_DATASET_STD,
        )

        self.images_dir = os.path.join(DATASET_DIR, 'images', dataset_type)

        with open(os.path.join(DATASET_DIR, 'wnids.txt')) as f:
            self.labels = [l.strip() for l in f]
            assert len(self.labels) == 1000, f'Not 1000 labels: {len(self.labels)}'

        self._wnid_to_label_idx = {
            wnid: idx
            for idx, wnid in enumerate(self.labels)
        }
        with open(os.path.join(DATASET_DIR, 'wnid_to_label.json')) as f:
            self._wnid_to_label_name = {
                k: t[0]
                for k, t in json.load(f).items()
            }

        # Load metadata
        fpath = os.path.join(DATASET_DIR, f'{dataset_type}_metadata.csv')
        self.metadata_df = pd.read_csv(fpath)

        # Mini
        if mini is not None:
            self.metadata_df = self.metadata_df.loc[self.metadata_df['mini'] == mini]

        if max_samples is not None:
            if max_samples <= len(self.metadata_df):
                # self.metadata_df = _keep_max_samples_balanced(
                #     self.metadata_df, self.labels, max_samples,
                # )
                self.metadata_df = self.metadata_df[:max_samples]

        self.metadata_df.reset_index(drop=True, inplace=True)


    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]

        image_fname = row['image_name']

        image_fpath = os.path.join(self.images_dir, image_fname)
        image = load_image(image_fpath, self.image_format)

        image = self.transform(image)

        wnid = self._wnid_to_label_idx[row['wnid']]

        return BatchItem(
            image=image,
            labels=wnid,
            image_fname=image_fname,
        )
