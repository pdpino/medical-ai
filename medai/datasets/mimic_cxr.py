import os
import re
import json
import logging
import torch
import pandas as pd
from torch.utils.data import Dataset

from medai.datasets.common import (
    BatchItem,
    CHEXPERT_DISEASES,
    JSRT_ORGANS,
    LATEST_REPORTS_VERSION,
    UP_TO_DATE_MASKS_VERSION,
)
from medai.datasets.vocab import load_vocab
from medai.utils.images import (
    get_default_image_transform,
    load_image,
    get_default_mask_transform,
)

LOGGER = logging.getLogger(__name__)

DATASET_DIR = os.environ.get('DATASET_DIR_MIMIC_CXR')
DATASET_DIR_FAST = os.environ.get('DATASET_DIR_MIMIC_CXR_FAST')

_REPORTS_FNAME = 'reports.clean.{}.json'

_DATASET_MEAN = 0.4719
_DATASET_STD = 0.3017

_FRONTAL_POSITIONS = ['PA', 'AP', 'AP AXIAL', 'LAO', 'LPO', 'RAO']

_BROKEN_IMAGES = set([
    'p11/p11285576/s54979966/03b2e67c-70631ff8-685825fb-6c989456-621ca64d.jpg',
    'p15/p15223781/s52459604/56b8afd3-5f6d4419-8699d79e-6913a2bd-35a08557.jpg',
    'p15/p15223781/s52459604/93020995-6b84ca33-2e41e00d-5d6e3bee-87cfe5c6.jpg',
    # Appears empty
    'p10/p10291098/s57194260/0539ee33-9d402e49-a9cc6d36-7aabc539-3d80a62b.jpg',
    # Blur empty images
    'p15/p15355458/s52423703/0b6f08b2-72deda00-d7ccc375-8278269f-b4e11c36.jpg',
    'p18/p18461911/s57183218/151abebe-2a750a5c-09c181bb-1a9016ef-92d8910e.jpg',
    'p19/p19839145/s54889255/f674e474-817bb713-8f16c90c-608cf869-2829cae7.jpg',
])

def _resolve_reports_version(version):
    if version == 'v4-1':
        # This version only changed in IU
        return 'v4'
    return version


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
                 do_not_load_image=False, do_not_load_report=False,
                 crop_center=None,
                 vocab=None, **unused_kwargs):
        super().__init__()

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

        self.seg_multilabel = seg_multilabel

        if image_size[0] <= 256 and image_size[1] <= 256:
            use_fast = True
            used_images_folder = 'images-small'
        else:
            use_fast = mini == 1
            used_images_folder = 'images'

        if not use_fast:
            LOGGER.warning('MIMIC loading images from HDD, will be slow')

        if not use_fast and DATASET_DIR is None:
            raise Exception('DATASET_DIR_MIMIC_CXR not found in env variables')
        if use_fast and DATASET_DIR_FAST is None:
            raise Exception('DATASET_DIR_MIMIC_CXR_FAST not found in env variables')

        self.images_dir = os.path.join(
            DATASET_DIR_FAST if use_fast else DATASET_DIR,
            used_images_folder,
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

        # Prepare reports for getter calls
        self._preprocess_reports(
            _resolve_reports_version(reports_version),
            studies=set(self.master_df['study_id']),
            vocab=vocab,
            vocab_greater=vocab_greater,
        )

        # Keep only studies from reports-version
        self.master_df = self.master_df.loc[self.master_df['study_id'].isin(set(self._reports))]

        # Sort samples
        if sort_samples:
            self.master_df = self.master_df.sort_values('report_length', ascending=True)

        # Keep only max images
        if max_samples is not None:
            self.master_df = self.master_df.tail(max_samples)

        # Reset the index, after all the modifications
        self.master_df.reset_index(drop=True, inplace=True)

        self.do_not_load_image = do_not_load_image
        self.do_not_load_report = do_not_load_report

        self.enable_masks = masks
        if masks:
            self.masks_dir = os.path.join(DATASET_DIR_FAST, 'masks', masks_version)

            assert os.path.isdir(self.masks_dir), f'Masks {self.masks_dir} do not exist'

            self.transform_mask = get_default_mask_transform(
                image_size,
                self.seg_multilabel,
                len(self.organs),
                crop_center=crop_center,
            )

        if not set(self.master_df['study_id']).issubset(self._reports):
            _missing_reports = set(self._reports) - set(self.master_df['study_id'])
            raise Exception(
                f'Not all reports from DF are processed! {len(_missing_reports)}',
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
        if self.do_not_load_report:
            tokens = -1
        else:
            report = self._reports[study_id]
            tokens = report['tokens_idxs']

        # Load image
        image = self._load_image(image_fpath)

        # Extract labels
        # pylint: disable=not-callable
        labels = torch.tensor(row[self.labels], dtype=torch.uint8)

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

    def _load_image(self, image_fpath):
        if self.do_not_load_image:
            # pylint: disable=not-callable
            return torch.tensor(-1)

        image_fpath = os.path.join(self.images_dir, image_fpath)
        image = load_image(image_fpath, self.image_format)

        image = self.transform(image)
        return image

    def load_masks(self, image_fpath):
        image_fpath = image_fpath.replace('/', '-').replace('.jpg', '.png')

        filepath = os.path.join(self.masks_dir, image_fpath)

        mask = load_image(filepath, 'L')
        mask = self.transform_mask(mask)
        return mask

    def _preprocess_reports(self, reports_version, studies, vocab=None, vocab_greater=None):
        # Load reports
        reports_fname = os.path.join(self.reports_dir, _REPORTS_FNAME.format(reports_version))
        with open(reports_fname, 'r') as f:
            reports_master_dict = json.load(f)

        if vocab is not None:
            self.word_to_idx = vocab
        else:
            self.word_to_idx = load_vocab(self.reports_dir, reports_version, vocab_greater)

        # Compute final reports array
        self._reports = dict()
        for study_id in studies:
            report = reports_master_dict[str(study_id)]

            clean_text = report['clean_text']
            tokens = clean_text.split()

            tokens_idxs = [
                self.word_to_idx[token]
                for token in tokens
                if token in self.word_to_idx
            ]
            self._reports[study_id] = {
                'clean_text': clean_text,
                'tokens_idxs': tokens_idxs,
            }

    def get_vocab(self):
        return self.word_to_idx

    def get_labels_presence_for(self, target_label):
        """Returns a list of tuples (idx, 0/1) indicating presence/absence of a
            label for each sample.
        """
        # FIXME: self._reports is by report-filename, not by image-filename!!
        if isinstance(target_label, int):
            target_label = self.labels[target_label]

        filename_to_label = dict(zip(
            self.labels_df['filename'],
            self.labels_df[target_label].astype(int),
        ))

        return [
            (index, filename_to_label[report['filename']])
            for index, report in enumerate(self._reports)
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
    def get_reports_by_id(self):
        return {
            idx: self._reports[study_id]['tokens_idxs']
            for idx, study_id in enumerate(self.master_df['study_id'])
        }
