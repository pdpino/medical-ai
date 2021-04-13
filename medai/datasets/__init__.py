import logging
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import default_collate

from medai.datasets.common import UP_TO_DATE_MASKS_VERSION
from medai.datasets.cxr14 import CXR14Dataset
from medai.datasets.chexpert import ChexpertDataset
from medai.datasets.covid_kaggle import CovidKaggleDataset
from medai.datasets.covid_x import CovidXDataset
from medai.datasets.covid_actual import CovidActualDataset
from medai.datasets.covid_fig1 import CovidFig1Dataset
from medai.datasets.covid_uc import CovidUCDataset
from medai.datasets.iu_xray import IUXRayDataset
from medai.datasets.mimic_cxr import MIMICCXRDataset
from medai.datasets.jsrt import JSRTDataset
from medai.datasets.vinbig import VinBigDataset

from medai.datasets.tools.oversampler import OneLabelOverSampler
from medai.datasets.tools.undersampler import OneLabelUnderSampler
from medai.datasets.tools.augmentation import Augmentator
from medai.datasets.tools.balanced_sampler import MultilabelBalancedSampler

from medai.utils.nlp import count_sentences

_CL_DATASETS = {
    'cxr14': CXR14Dataset,
    'chexpert': ChexpertDataset,
    'covid-kaggle': CovidKaggleDataset,
    'covid-x': CovidXDataset,
    'covid-actual': CovidActualDataset,
    'covid-fig1': CovidFig1Dataset,
    'covid-uc': CovidUCDataset,
    'iu-x-ray': IUXRayDataset,
    'vinbig': VinBigDataset,
}

_RG_DATASETS = {
    'iu-x-ray': IUXRayDataset,
    'mimic-cxr': MIMICCXRDataset,
}

_SEG_DATASETS = {
    'jsrt': JSRTDataset,
    'vinbig': VinBigDataset,
}

AVAILABLE_CLASSIFICATION_DATASETS = list(_CL_DATASETS)
AVAILABLE_REPORT_DATASETS = list(_RG_DATASETS)
AVAILABLE_SEGMENTATION_DATASETS = list(_SEG_DATASETS)

LOGGER = logging.getLogger(__name__)


_MISSING_SPLITS = set([
    ('chexpert', 'test'),
])

_DATASETS_WITH_MASKS_IMPLEMENTED = set([
    'cxr14', 'iu-x-ray', 'vinbig', 'chexpert',
])


def _classification_collate_fn(batch_items):
    batch_items = [
        batch_item._replace(report=-1)
        for batch_item in batch_items
    ]
    return default_collate(batch_items)


def prepare_data_classification(dataset_name='cxr14', dataset_type='train',
                                labels=None,
                                max_samples=None, image_size=(512, 512),
                                augment=False, augment_mode='single',
                                augment_seg_mask=False,
                                augment_label=None, augment_class=None, augment_times=1,
                                augment_kwargs={},
                                oversample=False,
                                oversample_label=0, oversample_class=None,
                                oversample_ratio=None, oversample_max_ratio=None,
                                undersample=False, undersample_label=0,
                                balanced_sampler=False,
                                batch_size=10, shuffle=False,
                                num_workers=2,
                                **kwargs,
                                ):
    if (dataset_name, dataset_type) in _MISSING_SPLITS:
        LOGGER.warning('Split is not available: %s/%s', dataset_name, dataset_type)
        return None

    if batch_size is None:
        LOGGER.warning('Loading data with batch_size=None')

    assert image_size is None or isinstance(image_size, (tuple, list)), (
        f'Image size must be a tuple, list, or None, got {image_size}'
    )

    assert dataset_name in _CL_DATASETS, f'Dataset not found: {dataset_name}'
    DatasetClass = _CL_DATASETS[dataset_name]

    if kwargs.get('masks', False):
        if dataset_name not in _DATASETS_WITH_MASKS_IMPLEMENTED:
            err = f'Dataset {dataset_name} does not have masks yet (masks=True)'
            raise NotImplementedError(err)

        if 'masks_version' not in kwargs:
            # backward compatibility: older runs do not have this set
            kwargs['masks_version'] = 'v0'

        masks_version_used = kwargs.get('masks_version', None)
        if masks_version_used != UP_TO_DATE_MASKS_VERSION:
            LOGGER.warning(
                'Not using the up-to-date masks_version, found=%s vs up-to-date=%s',
                masks_version_used, UP_TO_DATE_MASKS_VERSION,
            )


    if kwargs.get('images_version') and dataset_name not in ('cxr14',):
        LOGGER.warning('images_version is not implemented in %s', dataset_name)
        kwargs['images_version'] = None


    _info = {
        'bs': batch_size,
        'imgsize': image_size,
        'version': kwargs.get('images_version'),
        'format': kwargs.get('image_format', 'RGB'),
        'n_labels': labels and len(labels),
    }
    _info_str = ' '.join(f'{k}={v}' for k, v in _info.items())
    LOGGER.info(
        'Loading %s/%s cl-dataset, %s', dataset_name, dataset_type, _info_str,
    )

    dataset = DatasetClass(dataset_type=dataset_type,
                           labels=labels,
                           image_size=image_size,
                           max_samples=max_samples,
                           **kwargs,
                           )

    if len(dataset) == 0:
        LOGGER.error('\tEmpty dataset')
        return None

    LOGGER.info('\tDataset size: %s', f'{len(dataset):,}')

    if augment:
        dataset = Augmentator(dataset,
                              label=augment_label,
                              force_class=augment_class,
                              mode=augment_mode,
                              times=augment_times,
                              seg_mask=augment_seg_mask,
                              **augment_kwargs)

    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'collate_fn': _classification_collate_fn,
    }

    if oversample:
        sampler = OneLabelOverSampler(dataset,
                                      label=oversample_label,
                                      force_class=oversample_class,
                                      ratio=oversample_ratio,
                                      max_ratio=oversample_max_ratio)
        dataloader_kwargs['sampler'] = sampler
    elif undersample:
        sampler = OneLabelUnderSampler(dataset, label=undersample_label)
        dataloader_kwargs['sampler'] = sampler
    elif balanced_sampler:
        if not dataset.multilabel:
            LOGGER.error('Balanced sampler only works for multilabel datasets, ignoring')
        else:
            dataloader_kwargs['sampler'] = MultilabelBalancedSampler(dataset)
    else:
        dataloader_kwargs['shuffle'] = shuffle

    dataloader = DataLoader(dataset, **dataloader_kwargs)
    return dataloader


def create_report_dataset_subset(dataset, max_n_words=None, max_n_sentences=None):
    """Creates a subset for a report dataset, considering only reports with a maximum
    length.

    Args:
        dataset: must contain reports attribute.
    """
    if max_n_words is None and max_n_sentences is None:
        return dataset

    stats = [
        (idx, len(report['tokens_idxs']), count_sentences(report['tokens_idxs']))
        for idx, report in enumerate(dataset.reports)
    ]

    if max_n_words is None:
        max_n_words = np.inf
    if max_n_sentences is None:
        max_n_sentences = np.inf

    indices = [
        idx
        for (idx, n_words, n_sentences) in stats
        if n_words <= max_n_words and n_sentences <= max_n_sentences
    ]

    return Subset(dataset, indices)


def prepare_data_report_generation(create_dataloader_fn,
                                   dataset_name=None,
                                   dataset_type='train',
                                   max_samples=None,
                                   vocab=None,
                                   image_size=(512, 512),
                                   batch_size=10,
                                   sort_samples=True,
                                   shuffle=False,
                                   augment=False, augment_mode='single',
                                   augment_label=None, augment_class=None,
                                   augment_seg_mask=False,
                                   augment_times=1, augment_kwargs={},
                                   num_workers=2,
                                   masks=False,
                                   **kwargs,
                                   ):

    _info = {
        'bs': batch_size,
        'imgsize': image_size,
    }
    _info_str = ' '.join(f'{k}={v}' for k, v in _info.items())
    LOGGER.info('Loading %s/%s rg-dataset, %s', dataset_name, dataset_type, _info_str)

    assert dataset_name in _RG_DATASETS, f'Dataset not found: {dataset_name}'
    DatasetClass = _RG_DATASETS[dataset_name]

    dataset = DatasetClass(dataset_type=dataset_type,
                           max_samples=max_samples,
                           vocab=vocab,
                           image_size=image_size,
                           sort_samples=sort_samples,
                           masks=masks,
                           **kwargs,
                           )

    if len(dataset) == 0:
        LOGGER.error('\tEmpty dataset')
        return None

    LOGGER.info('\tDataset size: %s', f'{len(dataset):,}')

    if augment:
        dataset = Augmentator(dataset,
                              label=augment_label,
                              force_class=augment_class,
                              mode=augment_mode,
                              times=augment_times,
                              seg_mask=augment_seg_mask,
                              **augment_kwargs)

    dataloader = create_dataloader_fn(dataset,
                                      include_masks=masks,
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      num_workers=num_workers,
                                      )

    return dataloader


def prepare_data_segmentation(dataset_name=None,
                              dataset_type='train',
                              image_size=(512, 512),
                              augment=False, augment_mode='single',
                              augment_label=None, augment_class=None, augment_times=1,
                              augment_kwargs={},
                              batch_size=10,
                              shuffle=False,
                              num_workers=2,
                              **kwargs,
                             ):
    LOGGER.info('Loading %s/%s seg-dataset, bs=%d...', dataset_name, dataset_type, batch_size)

    assert dataset_name in _SEG_DATASETS, f'Dataset not found: {dataset_name}'
    DatasetClass = _SEG_DATASETS[dataset_name]

    dataset = DatasetClass(dataset_type=dataset_type,
                           image_size=image_size,
                           **kwargs,
                          )

    if len(dataset) == 0:
        LOGGER.error('\tEmpty dataset')
        return None

    LOGGER.info('\tDataset size: %s', f'{len(dataset):,}')

    if augment:
        dataset = Augmentator(dataset,
                              label=augment_label,
                              force_class=augment_class,
                              mode=augment_mode,
                              times=augment_times,
                              seg_mask=True,
                              **augment_kwargs)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                           )
    return dataloader
