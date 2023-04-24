import os
import json
import logging
import pandas as pd
import torch
from torch.utils.data import Dataset

from medai.datasets.common import (
    BatchItem,
    CHEXPERT_LABELS,
    COATT_LABELS,
    IU_MTI_TAGS,
    JSRT_ORGANS,
    UP_TO_DATE_MASKS_VERSION,
    LATEST_REPORTS_VERSION,
)
from medai.datasets.vocab import load_vocab
from medai.utils.images import (
    get_default_image_transform,
    load_image,
    get_default_mask_transform,
)


LOGGER = logging.getLogger(__name__)

DATASET_DIR = os.environ.get('DATASET_DIR_IU_XRAY')
_REPORTS_FNAME = 'reports.clean.{}.json'

_AVAILABLE_SPLITS = ['train', 'val', 'test', 'all']

_DATASET_MEAN = 0.4821
_DATASET_STD = 0.2374

def _get_reports_names_from_split(dataset_type):
    list_fname = os.path.join(DATASET_DIR, 'splits', f'{dataset_type}.txt')
    with open(list_fname, 'r') as f:
        reports_from_split = [l.strip() for l in f.readlines()]
    return reports_from_split


class IUXRayDataset(Dataset):
    dataset_name = 'iu-x-ray'
    dataset_dir = DATASET_DIR

    organs = list(JSRT_ORGANS)

    def __init__(self, dataset_type='train', max_samples=None,
                 labels=None,
                 sort_samples=True,
                 frontal_only=False, image_size=(512, 512),
                 norm_by_sample=False,
                 image_format='RGB',
                 masks=False, masks_version=UP_TO_DATE_MASKS_VERSION,
                 seg_multilabel=True, reports_version=LATEST_REPORTS_VERSION,
                 vocab_greater=None,
                 do_not_load_image=False, do_not_load_report=False,
                 crop_center=None,
                 vocab=None, **unused_kwargs):
        super().__init__()

        if DATASET_DIR is None:
            raise Exception('DATASET_DIR_IU_XRAY not found in env variables')

        if dataset_type not in _AVAILABLE_SPLITS:
            raise ValueError(f'No such type, must be in {_AVAILABLE_SPLITS}')

        self.dataset_type = dataset_type
        self.image_format = image_format
        self.image_size = image_size if crop_center is None else (crop_center, crop_center)
        self.transform = get_default_image_transform(
            self.image_size,
            norm_by_sample=norm_by_sample,
            crop_center=crop_center,
            mean=_DATASET_MEAN,
            std=_DATASET_STD,
        )

        self.images_dir = os.path.join(DATASET_DIR, 'images')
        self.reports_dir = os.path.join(DATASET_DIR, 'reports')
        self.dataset_pdpino_dir = os.path.join(DATASET_DIR).replace('dataset', 'dataset-pdpino')

        # Only frontal masks are available
        assert not masks or frontal_only, 'if masks is True, set frontal_only=True'

        self.multilabel = True # CL multilabel
        self.seg_multilabel = seg_multilabel
        self._preprocess_labels(labels)

        self.enable_masks = masks

        # Load reports
        self.reports_version = reports_version
        reports_fname = os.path.join(self.dataset_pdpino_dir, _REPORTS_FNAME.format(reports_version))
        with open(reports_fname, 'r') as f:
            reports = list(json.load(f).values())

        # Filter by train, val, test
        if dataset_type == 'all':
            splits = ('train', 'val', 'test')
        else:
            splits = (dataset_type,)

        reports_from_split = []
        for split in splits:
            reports_from_split.extend(_get_reports_names_from_split(split))
        reports = [rep for rep in reports if rep['filename'] in reports_from_split]

        # Prepare reports for getter calls
        self._preprocess_reports(reports, sort_samples=sort_samples,
                                 vocab=vocab,
                                 vocab_greater=vocab_greater,
                                 frontal_only=frontal_only)

        # Keep only max images
        if max_samples is not None:
            last_samples = -1 * max_samples
            self.samples = self.samples[last_samples:]

        if self.enable_masks:
            self.masks_dir = os.path.join(DATASET_DIR, 'masks', masks_version)
            assert os.path.isdir(self.masks_dir), f'Masks {self.masks_dir} do not exist'

            self.transform_mask = get_default_mask_transform(
                image_size,
                self.seg_multilabel,
                len(self.organs),
                crop_center=crop_center,
            )

        self.do_not_load_image = do_not_load_image
        self.do_not_load_report = do_not_load_report

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        report_fname = sample['report_filename']
        image_fname = sample['image_name']
        image = self._load_image(image_fname)

        labels = self.labels_by_report[report_fname]

        mask = self.load_mask(image_fname) if self.enable_masks else -1

        report = sample['tokens_idxs'] if not self.do_not_load_report else -1

        return BatchItem(
            image=image,
            labels=labels,
            report=report,
            image_fname=image_fname,
            report_fname=report_fname,
            masks=mask,
        )

    def _load_image(self, image_name):
        if self.do_not_load_image:
            # pylint: disable=not-callable
            return torch.tensor(-1)

        image_path = os.path.join(self.images_dir, f'{image_name}.png')
        image = load_image(image_path, self.image_format)

        image = self.transform(image)
        return image

    def load_mask(self, image_name):
        filepath = os.path.join(self.masks_dir, f'{image_name}.png')

        mask = load_image(filepath, 'L')
        mask = self.transform_mask(mask)
        return mask

    def get_vocab(self):
        return self.word_to_idx

    def _preprocess_reports(self, reports, sort_samples=True, vocab=None, vocab_greater=None,
                            frontal_only=False):
        if vocab is not None:
            self.word_to_idx = vocab
        else:
            self.word_to_idx = load_vocab(self.reports_dir, self.reports_version, vocab_greater)

        self.n_unique_reports = len(reports)

        # Compute final reports array
        self.samples = []
        for report in reports:
            filename = report['filename']

            tokens = report['clean_text'].split()

            tokens_idxs = [
                self.word_to_idx[token]
                for token in tokens
                if token in self.word_to_idx
            ]

            for image in report['images']:
                position = image['side']
                if frontal_only and 'frontal' not in position:
                    continue

                if image['broken']:
                    continue

                self.samples.append({
                    'report_filename': filename,
                    'tokens_idxs': tokens_idxs,
                    'image_name': image['id'],
                    # 'position': position, # Use this for debugging
                })

        if sort_samples:
            self.samples = sorted(self.samples, key=lambda x:len(x['tokens_idxs']))

    def _preprocess_labels(self, labels=None):
        # Choose labels to use
        load_from = 'chexpert'

        if labels is None:
            self.labels = list(CHEXPERT_LABELS)
        elif labels == 'coatt-labels' or labels == ['coatt-labels']:
            # HACKy way to solve this!
            self.labels = list(COATT_LABELS)
            load_from = 'coatt'
        elif labels == 'mti' or labels == ['mti']:
            self.labels = list(IU_MTI_TAGS)
            load_from = 'mti'
        else:
            self.labels = [l for l in labels if l in CHEXPERT_LABELS]

        if load_from == 'chexpert':
            # Load Dataframe
            path = os.path.join(self.reports_dir,
                                'reports_with_chexpert_labels.csv')
            self.labels_df = pd.read_csv(path)

            # Transform uncertains and none to 0
            self.labels_df = self.labels_df.replace({
                -1: 1, # uncertain values, assumed positive
                -2: 0, # No mention, assumed negative
            })
        elif load_from == 'coatt':
            path = os.path.join(DATASET_DIR, 'coatt-labels',
                                'metadata.csv')
            self.labels_df = pd.read_csv(path)
        elif load_from == 'mti':
            path = os.path.join(DATASET_DIR, 'mti-tags.csv')
            self.labels_df = pd.read_csv(path)

        # Save in a more convenient storage for __getitem__
        self.labels_by_report = dict()
        for _, row in self.labels_df.iterrows():
            filename = row['filename']
            # pylint: disable=not-callable
            labels = torch.tensor(row[self.labels], dtype=torch.uint8)

            # # If all labels are 0 --> set no-findings==1
            # # (notice no-findings and support-devices are ignored)
            # if labels[1:-1].sum() == 0 and labels[0] == 0:
            #     labels[0] = 1

            self.labels_by_report[filename] = labels

    def get_labels_presence_for(self, target_label):
        """Returns a list of tuples (idx, 0/1) indicating presence/absence of a
            label for each sample.
        """
        # FIXME: labels_df is by report-filename, not by image-filename!!
        if isinstance(target_label, int):
            target_label = self.labels[target_label]

        filename_to_label = dict(zip(
            self.labels_df['filename'],
            self.labels_df[target_label].astype(int),
        ))

        return [
            (index, filename_to_label[report['filename']])
            for index, report in enumerate(self.samples)
        ]

    def get_presence_for_no_finding(self):
        return self.get_labels_presence_for('No Finding')

    ### API for dummy models
    def get_reports_by_id(self):
        return {
            sample['image_name']: sample['tokens_idxs']
            for sample in self.samples
        }
