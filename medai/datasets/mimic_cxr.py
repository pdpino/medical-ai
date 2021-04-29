import os
import re
import json
import logging
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from ignite.utils import to_onehot

from medai.datasets.common import (
    BatchItem,
    CHEXPERT_DISEASES,
    JSRT_ORGANS,
    LATEST_REPORTS_VERSION,
    UP_TO_DATE_MASKS_VERSION,
)
from medai.datasets.vocab import load_vocab
from medai.utils.images import get_default_image_transform

LOGGER = logging.getLogger(__name__)

DATASET_DIR = os.environ.get('DATASET_DIR_MIMIC_CXR')
DATASET_DIR_FAST = os.environ.get('DATASET_DIR_MIMIC_CXR_FAST')

_REPORTS_FNAME = 'reports.clean.{}.json'

_DATASET_MEAN = 0.4719
_DATASET_STD = 0.3017

_FRONTAL_POSITIONS = ['PA', 'AP', 'AP AXIAL', 'LAO', 'LPO', 'RAO']

_BROKEN_IMAGES = set([
    # Appears empty
    'p10/p10291098/s57194260/0539ee33-9d402e49-a9cc6d36-7aabc539-3d80a62b.jpg',
])

class MIMICCXRDataset(Dataset):
    organs = list(JSRT_ORGANS)
    dataset_name = 'mimic-cxr'

    dataset_dir = DATASET_DIR
    dataset_dir_fast = DATASET_DIR_FAST
    multilabel = True

    def __init__(self, dataset_type='train', max_samples=None,
                 labels=None, image_size=(512, 512),
                 norm_by_sample=False,
                 image_format='RGB',
                 sort_samples=False, frontal_only=False,
                 mini=None,
                 masks=False, masks_version=UP_TO_DATE_MASKS_VERSION,
                 seg_multilabel=True,
                 vocab_greater=None, reports_version=LATEST_REPORTS_VERSION,
                 do_not_load_image=False,
                 vocab=None, **unused_kwargs):
        super().__init__()

        if DATASET_DIR is None:
            raise Exception('DATASET_DIR_MIMIC_CXR not found in env variables')
        if mini == 1 and DATASET_DIR_FAST is None:
            raise Exception('DATASET_DIR_MIMIC_CXR_FAST not found in env variables')

        self.dataset_type = dataset_type
        self.image_format = image_format
        self.image_size = image_size
        self.transform = get_default_image_transform(
            self.image_size,
            norm_by_sample=norm_by_sample,
            mean=_DATASET_MEAN,
            std=_DATASET_STD,
        )

        self.seg_multilabel = seg_multilabel

        self.images_dir = os.path.join(
            DATASET_DIR_FAST if mini == 1 else DATASET_DIR,
            'images',
        )
        self.reports_dir = os.path.join(DATASET_DIR, 'reports')

        # Save labels
        if labels is not None:
            self.labels = list(labels)
        else:
            self.labels = CHEXPERT_DISEASES

        # Load master_df
        fpath = os.path.join(DATASET_DIR, 'master_metadata.csv')
        self.master_df = pd.read_csv(fpath)

        # Filter by train, val, test
        if dataset_type != 'all':
            available_splits = list(self.master_df['split'].unique()) + ['all']
            self.master_df = self.master_df.loc[self.master_df['split'] == dataset_type]
            if len(self.master_df) == 0:
                raise Exception(f'{dataset_type} split not available, only {available_splits}')

        if frontal_only:
            df = self.master_df
            self.master_df = df.loc[df['ViewPosition'].isin(_FRONTAL_POSITIONS)]

        if mini is not None:
            df = self.master_df
            self.master_df = df.loc[df['mini'] == mini]
        self._mini = mini

        # Ignore broken images
        self.master_df = self.master_df.loc[~self.master_df['image_fpath'].isin(_BROKEN_IMAGES)]

        # Keep only max images
        if max_samples is not None:
            self.master_df = self.master_df.tail(max_samples)

        if sort_samples:
            self.master_df = self.master_df.sort_values('report_length', ascending=True)
        self.master_df.reset_index(drop=True, inplace=True)

        # Prepare reports for getter calls
        self._preprocess_reports(
            reports_version,
            studies=set(self.master_df['study_id']),
            vocab=vocab,
            vocab_greater=vocab_greater,
        )

        self.do_not_load_image = do_not_load_image

        self.enable_masks = masks
        if masks:
            self.masks_dir = os.path.join(DATASET_DIR_FAST, 'masks', masks_version)

            assert os.path.isdir(self.masks_dir), f'Masks {masks_version} not calculated!'

            self.transform_mask = transforms.Compose([
                transforms.Resize(image_size, 0), # Nearest mode
                transforms.ToTensor(),
            ])


    def __len__(self):
        return len(self.master_df)

    def __getitem__(self, idx):
        row = self.master_df.iloc[idx]

        # Extract names
        image_fpath = row['image_fpath']
        report_fpath = row['report_fpath']
        study_id = row['study_id']

        # Extract report
        report = self.reports[study_id]
        tokens = report['tokens_idxs']

        # Load image
        image = self.load_image(image_fpath)

        # Extract labels
        labels = torch.ByteTensor(row[self.labels])

        # Load masks
        masks = self.load_masks(image_fpath) if self.enable_masks else -1

        return BatchItem(
            image=image,
            labels=labels,
            report=tokens,
            masks=masks,
            image_fname=image_fpath,
            report_fname=report_fpath,
        )

    def load_image(self, image_fpath):
        if self.do_not_load_image:
            # pylint: disable=not-callable
            return torch.tensor(-1)

        image_fpath = os.path.join(self.images_dir, image_fpath)
        try:
            image = Image.open(image_fpath).convert(self.image_format)
        except OSError as e:
            LOGGER.error(
                '%s: Failed to load image, may be broken: %s',
                self.dataset_type, image_fpath,
            )
            LOGGER.error(e)
            raise

        image = self.transform(image)
        return image

    def load_masks(self, image_fpath):
        image_fpath = image_fpath.replace('/', '-').replace('.jpg', '.png')

        filepath = os.path.join(self.masks_dir, image_fpath)

        if not os.path.isfile(filepath):
            LOGGER.error('No such mask: %s', filepath)
            return None

        mask = Image.open(filepath).convert('L')
        mask = self.transform_mask(mask)
        # shape: n_channels=1, height, width

        mask = (mask * 255).long()
        # shape: 1, height, width

        if self.seg_multilabel:
            mask = to_onehot(mask, len(self.organs))
            # shape: 1, n_organs, height, width

        mask = mask.squeeze(0)
        # shape(seg_multilabel=True): n_organs, height, width
        # shape(seg_multilabel=False): height, width

        return mask

    def _preprocess_reports(self, reports_version, studies, vocab=None, vocab_greater=None):
        # Load reports
        reports_fname = os.path.join(self.reports_dir, _REPORTS_FNAME.format(reports_version))
        with open(reports_fname, 'r') as f:
            reports_master_dict = json.load(f)

        if vocab is not None:
            self.word_to_idx = vocab
        else:
            self.word_to_idx = load_vocab('mimic_cxr', vocab_greater)

        # Compute final reports array
        self.reports = dict()
        for study_id in studies:
            report = reports_master_dict[str(study_id)]

            clean_text = report['clean_text']
            tokens = clean_text.split()

            tokens_idxs = [
                self.word_to_idx[token]
                for token in tokens
                if token in self.word_to_idx
            ]
            self.reports[study_id] = {
                'clean_text': clean_text,
                'tokens_idxs': tokens_idxs,
            }

    def get_vocab(self):
        return self.word_to_idx

    def get_labels_presence_for(self, target_label):
        """Returns a list of tuples (idx, 0/1) indicating presence/absence of a
            label for each sample.
        """
        if isinstance(target_label, int):
            target_label = self.labels[target_label]

        filename_to_label = dict(zip(
            self.labels_df['filename'],
            self.labels_df[target_label].astype(int),
        ))

        return [
            (index, filename_to_label[report['filename']])
            for index, report in enumerate(self.reports)
        ]

    def get_presence_for_no_finding(self):
        return self.get_labels_presence_for('No Finding')

    def image_names_to_indexes(self, image_names):
        _clean_name_regex = re.compile(r'(p\d{2})-(p\d+)-(s\d+)-([\d\-\.\w]+)')
        def _clean_name(name):
            found = _clean_name_regex.search(name)
            if found:
                name = '/'.join(found.group(g) for g in (1, 2, 3, 4))
            name = name.replace('.png', '')
            if not name.endswith('.jpg'):
                name = f'{name}.jpg'
            return name

        if isinstance(image_names, str):
            image_names = (image_names,)
        image_names = set(
            _clean_name(name)
            for name in image_names
        )

        rows = self.master_df.loc[self.master_df['image_fpath'].isin(image_names)]
        return rows.index

    ### API for dummy models
    def iter_reports_only(self):
        return self.reports.values()
