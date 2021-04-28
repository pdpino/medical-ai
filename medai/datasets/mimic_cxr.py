import os
import json
import logging
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from medai.datasets.common import BatchItem, CHEXPERT_DISEASES, LATEST_REPORTS_VERSION
from medai.datasets.vocab import load_vocab
from medai.utils.images import get_default_image_transform

LOGGER = logging.getLogger(__name__)

DATASET_DIR = os.environ.get('DATASET_DIR_MIMIC_CXR')
DATASET_DIR_FAST = os.environ.get('DATASET_DIR_MIMIC_CXR_FAST')

_REPORTS_FNAME = 'reports.clean.{}.json'

_DATASET_MEAN = 0.4719
_DATASET_STD = 0.3017

_FRONTAL_POSITIONS = ['PA', 'AP', 'AP AXIAL', 'LAO', 'LPO', 'RAO']

class MIMICCXRDataset(Dataset):
    dataset_dir = DATASET_DIR
    multilabel = True
    enable_masks = False

    def __init__(self, dataset_type='train', max_samples=None,
                 labels=None, image_size=(512, 512),
                 norm_by_sample=False,
                 image_format='RGB',
                 sort_samples=False, frontal_only=False,
                 mini=None,
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

        # Keep only max images
        if max_samples is not None:
            self.master_df = self.master_df.tail(max_samples)

        if sort_samples:
            self.master_df = self.master_df.sort_values('report_length', ascending=True)
            self.master_df.reset_index(drop=True, inplace=True)

        # Prepare reports for getter calls
        self._preprocess_reports(
            reports_version,
            vocab=vocab,
            vocab_greater=vocab_greater,
        )

        self.do_not_load_image = do_not_load_image

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

        return BatchItem(
            image=image,
            labels=labels,
            report=tokens,
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

    def get_vocab(self):
        return self.word_to_idx

    def _preprocess_reports(self, reports_version, vocab=None, vocab_greater=None):
        # Load reports
        reports_fname = os.path.join(self.reports_dir, _REPORTS_FNAME.format(reports_version))
        with open(reports_fname, 'r') as f:
            reports = list(json.load(f).values())

        if vocab is not None:
            self.word_to_idx = vocab
        else:
            self.word_to_idx = load_vocab('mimic_cxr', vocab_greater)

        # Compute final reports array
        self.reports = dict()
        for report in reports:
            study_id = report['study_id']

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


    ### API for dummy models
    def iter_reports_only(self):
        return self.reports.values()
