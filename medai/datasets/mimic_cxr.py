import os
import json
import logging
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from medai.datasets.common import BatchItem, CHEXPERT_DISEASES
from medai.datasets.vocab import load_vocab, compute_vocab
from medai.utils.images import get_default_image_transform
from medai.utils.nlp import (
    UNKNOWN_IDX,
)

LOGGER = logging.getLogger(__name__)

DATASET_DIR = os.environ.get('DATASET_DIR_MIMIC_CXR')

_REPORTS_FNAME = 'reports.clean.v1.json'

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
                 sort_samples=False,
                 vocab=None, recompute_vocab=False, **unused_kwargs):
        super().__init__()

        if DATASET_DIR is None:
            raise Exception('DATASET_DIR_MIMIC_CXR not found in env variables')

        self.dataset_type = dataset_type
        self.image_format = image_format
        self.image_size = image_size
        self.transform = get_default_image_transform(
            self.image_size,
            norm_by_sample=norm_by_sample,
            mean=_DATASET_MEAN,
            std=_DATASET_STD,
        )

        self.images_dir = os.path.join(DATASET_DIR, 'images')
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

        # Keep only max images
        if max_samples is not None:
            self.master_df = self.master_df.tail(max_samples)

        if sort_samples:
            self.master_df = self.master_df.sort_values('report_length', ascending=True)
            self.master_df.reset_index(drop=True, inplace=True)

        # Prepare reports for getter calls
        self._preprocess_reports(
            vocab=vocab,
            recompute_vocab=recompute_vocab,
        )

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
        tokens = report['tokens']

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

    def _preprocess_reports(self, vocab=None,
                            recompute_vocab=False):
        # Load reports
        reports_fname = os.path.join(self.reports_dir, _REPORTS_FNAME)
        with open(reports_fname, 'r') as f:
            reports = list(json.load(f).values())

        if recompute_vocab:
            self.word_to_idx = compute_vocab(
                report['clean_text'].split()
                for report in reports
            )
        elif vocab is not None:
            self.word_to_idx = vocab
        else:
            self.word_to_idx = load_vocab('mimic_cxr')

        # n_unique_reports = len(reports)

        # Compute final reports array
        self.reports = dict()
        for report in reports:
            study_id = report['study_id']

            clean_text = report['clean_text']
            tokens = clean_text.split()

            tokens_idxs = [
                self.word_to_idx.get(token, UNKNOWN_IDX)
                for token in tokens
            ]
            self.reports[study_id] = {
                'clean_text': clean_text,
                'tokens': tokens_idxs,
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
