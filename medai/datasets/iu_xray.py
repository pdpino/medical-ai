import os
import json
import logging
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from ignite.utils import to_onehot

from medai.datasets.common import (
    BatchItem,
    CHEXPERT_LABELS,
    JSRT_ORGANS,
    UP_TO_DATE_MASKS_VERSION,
)
from medai.datasets.vocab import load_vocab, compute_vocab
from medai.utils.images import get_default_image_transform
from medai.utils.nlp import (
    UNKNOWN_IDX,
)

LOGGER = logging.getLogger(__name__)

DATASET_DIR = os.environ.get('DATASET_DIR_IU_XRAY')
_REPORTS_FNAME = 'reports.clean.v2.json'

_AVAILABLE_SPLITS = ['train', 'val', 'test', 'all']

_DATASET_MEAN = 0.4821
_DATASET_STD = 0.2374


class IUXRayDataset(Dataset):
    organs = list(JSRT_ORGANS)
    dataset_dir = DATASET_DIR

    def __init__(self, dataset_type='train', max_samples=None,
                 labels=None,
                 sort_samples=True,
                 frontal_only=False, image_size=(512, 512),
                 norm_by_sample=False,
                 image_format='RGB',
                 masks=False, masks_version=UP_TO_DATE_MASKS_VERSION,
                 seg_multilabel=True,
                 vocab=None, recompute_vocab=False, **unused_kwargs):
        super().__init__()

        if DATASET_DIR is None:
            raise Exception('DATASET_DIR_IU_XRAY not found in env variables')

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

        self.enable_masks = masks
        self.transform_mask = transforms.Compose([
            transforms.Resize(image_size, 0), # Nearest mode
            transforms.ToTensor(),
        ])

        self.multilabel = True # CL multilabel
        self.seg_multilabel = seg_multilabel
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

        # Prepare reports for getter calls
        self._preprocess_reports(reports, sort_samples=sort_samples,
                                 vocab=vocab, recompute_vocab=recompute_vocab,
                                 frontal_only=frontal_only)

        # Keep only max images
        if max_samples is not None:
            self.reports = self.reports[:max_samples]

        if self.enable_masks:
            self.masks_dir = os.path.join(DATASET_DIR, 'masks', masks_version)
            assert os.path.isdir(self.masks_dir), f'Masks {masks_version} not calculated!'


    def __len__(self):
        return len(self.reports)

    def __getitem__(self, idx):
        report = self.reports[idx]

        report_fname = report['filename']
        image_fname = report['image_name']
        image = self.load_image(image_fname)

        labels = self.labels_by_report[report_fname]

        mask = self.load_mask(image_fname) if self.enable_masks else -1

        return BatchItem(
            image=image,
            labels=labels,
            report=report['tokens_idxs'],
            image_fname=image_fname,
            report_fname=report_fname,
            masks=mask,
            )

    def load_image(self, image_name):
        image_path = os.path.join(self.images_dir, f'{image_name}.png')
        try:
            image = Image.open(image_path).convert(self.image_format)
        except OSError as e:
            LOGGER.error(
                '%s: Failed to load image, may be broken: %s',
                self.dataset_type, image_path,
            )
            LOGGER.error(e)
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

        if self.seg_multilabel:
            mask = to_onehot(mask, len(self.organs))
            # shape: 1, n_organs, height, width

        mask = mask.squeeze(0)
        # shape(seg_multilabel=True): n_organs, height, width
        # shape(seg_multilabel=False): height, width

        return mask

    def get_vocab(self):
        return self.word_to_idx

    def _preprocess_reports(self, reports, sort_samples=True, vocab=None,
                            recompute_vocab=False, frontal_only=False):
        if recompute_vocab:
            self.word_to_idx = compute_vocab(
                report['clean_text'].split()
                for report in reports
            )
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
        for _, row in self.labels_df.iterrows():
            filename = row['filename']
            labels = row[self.labels].to_numpy().astype(int)

            # # If all labels are 0 --> set no-findings==1
            # # (notice no-findings and support-devices are ignored)
            # if labels[1:-1].sum() == 0 and labels[0] == 0:
            #     labels[0] = 1

            # pylint: disable=not-callable
            self.labels_by_report[filename] = torch.tensor(labels)

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
