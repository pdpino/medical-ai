import torch
from torch.utils.data import Dataset
from torchvision import transforms
from ignite.utils import to_onehot
from PIL import Image
import os
import json
import pandas as pd

from medai.datasets.common import BatchItem, CHEXPERT_LABELS, JSRT_ORGANS
from medai.datasets.vocab import load_vocab
from medai.utils.images import get_default_image_transform
from medai.utils.nlp import (
    UNKNOWN_IDX,
    compute_vocab,
)

DATASET_DIR = os.environ.get('DATASET_DIR_IU_XRAY')
_REPORTS_FNAME = 'reports.clean.v2.json'

_AVAILABLE_SPLITS = ['train', 'val', 'test', 'all']

def _reports_iterator(reports):
    for report in reports:
        yield report['clean_text'].split()

_DATASET_MEAN = 0.4821
_DATASET_STD = 0.2374


class IUXRayDataset(Dataset):
    def __init__(self, dataset_type='train', max_samples=None,
                 labels=None,
                 sort_samples=True,
                 frontal_only=False, image_size=(512, 512),
                 norm_by_sample=False,
                 image_format='RGB',
                 masks=False,
                 vocab=None, recompute_vocab=False, **unused):
        if DATASET_DIR is None:
            raise Exception(f'DATASET_DIR_IU_XRAY not found in env variables')

        if dataset_type not in _AVAILABLE_SPLITS:
            raise ValueError(f'No such type, must be in {_AVAILABLE_SPLITS}')

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

        # Only frontal masks are available
        assert not masks or frontal_only, 'if masks is True, set frontal_only=True'

        self.masks_dir = os.path.join(DATASET_DIR, 'masks')
        self.enable_masks = masks
        self.transform_mask = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        if self.enable_masks:
            fpath = os.path.join(self.reports_dir, 'sentences_with_organs.csv')
            self.organs_by_sentence_df = pd.read_csv(fpath)
            self.organs = list(JSRT_ORGANS)
        else:
            self.organs_by_sentence_df = None
            self.organs = None

        self.multilabel = True
        self._preprocess_labels(labels)

        # Load reports
        reports_fname = os.path.join(self.reports_dir, _REPORTS_FNAME)
        with open(reports_fname, 'r') as f:
            reports = list(json.load(f).values())

        # Filter by train, val, test
        if dataset_type != 'all':
            list_fname = os.path.join(self.reports_dir, f'{dataset_type}.txt')
            with open(list_fname, 'r') as f:
                reports_from_split = set(l.strip() for l in f.readlines())
            reports = [rep for rep in reports if rep['filename'] in reports_from_split]

        # Keep only max images
        if max_samples is not None:
            reports = reports[:max_samples]

        # Save amounts
        self._preprocess_reports(reports, sort_samples=sort_samples,
                                 vocab=vocab, recompute_vocab=recompute_vocab,
                                 frontal_only=frontal_only)

    def __len__(self):
        return len(self.reports)

    def __getitem__(self, idx):
        report = self.reports[idx]

        report_filename = report['filename']
        image_name = report['image_name']
        image = self.load_image(image_name)

        labels = self.labels_by_report[report_filename]

        mask = self.load_mask(image_name) if self.enable_masks else None

        return BatchItem(
            image=image,
            labels=labels,
            report=report['tokens_idxs'],
            filename=image_name,
            masks=mask,
            )

    def load_image(self, image_name):
        image_path = os.path.join(self.images_dir, f'{image_name}.png')
        try:
            image = Image.open(image_path).convert(self.image_format)
        except OSError as e:
            print(f'({self.dataset_type}) Failed to load image, may be broken: {image_path}')
            print(e)
            raise

        image = self.transform(image)
        return image

    def load_mask(self, image_name):
        filepath = os.path.join(self.masks_dir, f'{image_name}.png')

        if not os.path.isfile(filepath):
            return None

        mask = Image.open(filepath).convert('L')
        mask = self.transform_mask(mask)
        # shape: n_channels=1, height, width

        mask = (mask * 255).long()
        # shape: 1, height, width

        mask = to_onehot(mask, len(self.organs))
        # shape: 1, n_organs, height, width

        mask = mask.squeeze(0)
        # shape: n_organs, height, width

        return mask

    def get_vocab(self):
        return self.word_to_idx

    def _preprocess_reports(self, reports, sort_samples=True, vocab=None,
                            recompute_vocab=False, frontal_only=False):
        if recompute_vocab:
            self.word_to_idx = compute_vocab(_reports_iterator(reports))
        elif vocab is not None:
            self.word_to_idx = vocab
        else:
            self.word_to_idx = load_vocab('iu_xray')

        self.n_unique_reports = len(reports)

        # Compute final reports array
        self.reports = []
        for report in reports:
            filename = report['filename']

            tokens = report['clean_text'].split()

            tokens_idxs = [
                self.word_to_idx.get(token, UNKNOWN_IDX)
                for token in tokens
            ]

            for image in report['images']:
                if frontal_only and 'frontal' not in image['side']:
                    continue

                if image['broken']:
                    continue

                self.reports.append({
                    'filename': filename,
                    'tokens_idxs': tokens_idxs,
                    'image_name': image['id'],
                })

        if sort_samples:
            self.reports = sorted(self.reports, key=lambda x:len(x['tokens_idxs']))

        # Reports are repeated to match n_images
        self.n_images = len(self.reports)

    def _preprocess_labels(self, labels=None):
        # Choose labels to use
        if labels is None:
            self.labels = list(CHEXPERT_LABELS)
        else:
            self.labels = [l for l in labels if l in CHEXPERT_LABELS]

        # Load Dataframe
        path = os.path.join(self.reports_dir,
                            'reports_with_chexpert_labels.csv')
        self.labels_df = pd.read_csv(path, index_col=0)

        # Transform uncertains and none to 0
        self.labels_df = self.labels_df.replace({
            -1: 1, # uncertain values, assumed positive
            -2: 0, # No mention, assumed negative
        })

        # Save in a more convenient storage for __getitem__
        self.labels_by_report = dict()
        for index, row in self.labels_df.iterrows():
            filename = row['filename']
            labels = row[self.labels].to_numpy().astype(int)

            # # If all labels are 0 --> set no-findings==1
            # # (notice no-findings and support-devices are ignored)
            # if labels[1:-1].sum() == 0 and labels[0] == 0:
            #     labels[0] = 1

            self.labels_by_report[filename] = torch.tensor(labels)

    def get_labels_presence_for(self, target_label):
        """Returns a list of tuples (idx, 0/1) indicating presence/absence of a
            label for each sample.
        """
        if isinstance(target_label, int):
            target_label = self.labels[target_label]

        filename_to_label = {
            filename: label
            for filename, label in zip(
                self.labels_df['filename'],
                self.labels_df[target_label].astype(int),
            )
        }

        return [
            (index, filename_to_label[report['filename']])
            for index, report in enumerate(self.reports)
        ]
