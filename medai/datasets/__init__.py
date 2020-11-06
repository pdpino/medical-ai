import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import default_collate

from medai.datasets.cxr14 import CXR14Dataset
from medai.datasets.covid_kaggle import CovidKaggleDataset
from medai.datasets.covid_x import CovidXDataset
from medai.datasets.covid_actual import CovidActualDataset
from medai.datasets.covid_fig1 import CovidFig1Dataset
from medai.datasets.covid_uc import CovidUCDataset
from medai.datasets.iu_xray import IUXRayDataset
from medai.datasets.jsrt import JSRTDataset

from medai.datasets.tools.oversampler import OneLabelOverSampler
from medai.datasets.tools.undersampler import OneLabelUnderSampler
from medai.datasets.tools.augmentation import Augmentator

from medai.utils.nlp import count_sentences

_CL_DATASETS = {
    'cxr14': CXR14Dataset,
    'covid-kaggle': CovidKaggleDataset,
    'covid-x': CovidXDataset,
    'covid-actual': CovidActualDataset,
    'covid-fig1': CovidFig1Dataset,
    'covid-uc': CovidUCDataset,
    'iu-x-ray': IUXRayDataset,
}

_RG_DATASETS = {
    'iu-x-ray': IUXRayDataset,
}

_SEG_DATASETS = {
    'jsrt': JSRTDataset,
}

AVAILABLE_CLASSIFICATION_DATASETS = list(_CL_DATASETS)
AVAILABLE_SEGMENTATION_DATASETS = list(_SEG_DATASETS)

def _classification_collate_fn(batch_items):
    batch_items = [
        batch_item._replace(report=-1)
        for batch_item in batch_items
    ]
    return default_collate(batch_items)


def prepare_data_classification(dataset_name='cxr14', dataset_type='train',
                                labels=None,
                                max_samples=None, image_size=(512, 512),
                                augment=False,
                                augment_label=None, augment_class=None, augment_kwargs={},
                                oversample=False,
                                oversample_label=0, oversample_class=None,
                                oversample_ratio=None, oversample_max_ratio=None,
                                undersample=False, undersample_label=0,
                                batch_size=10, shuffle=False,
                                num_workers=2,
                                **kwargs,
                                ):
    print(f'Loading {dataset_name}/{dataset_type} dataset...')

    assert dataset_name in _CL_DATASETS, f'Dataset not found: {dataset_name}'
    DatasetClass = _CL_DATASETS[dataset_name]

    dataset = DatasetClass(dataset_type=dataset_type,
                           labels=labels,
                           image_size=image_size,
                           max_samples=max_samples,
                           **kwargs,
                           )

    if len(dataset) == 0:
        return None

    if augment:
        dataset = Augmentator(dataset,
                              label=augment_label,
                              force_class=augment_class,
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
        dataloader = DataLoader(dataset, sampler=sampler, **dataloader_kwargs)
    elif undersample:
        sampler = OneLabelUnderSampler(dataset, label=undersample_label)
        dataloader = DataLoader(dataset, sampler=sampler, **dataloader_kwargs)
    else:
        dataloader = DataLoader(dataset, shuffle=shuffle, **dataloader_kwargs)

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
                                   augment=False,
                                   augment_label=None, augment_class=None, augment_kwargs={},
                                   num_workers=2):
    print(f'Loading {dataset_name}/{dataset_type} dataset...')

    assert dataset_name in _RG_DATASETS, f'Dataset not found: {dataset_name}'
    DatasetClass = _RG_DATASETS[dataset_name]

    dataset = DatasetClass(dataset_type=dataset_type,
                           max_samples=max_samples,
                           vocab=vocab,
                           image_size=image_size,
                           sort_samples=sort_samples,
                           )

    if augment:
        dataset = Augmentator(dataset,
                              label=augment_label,
                              force_class=augment_class,
                              **augment_kwargs)

    dataloader = create_dataloader_fn(dataset,
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      num_workers=num_workers,
                                      )

    return dataloader


def prepare_data_segmentation(dataset_name=None,
                              dataset_type='train',
                              image_size=(512, 512),
                              batch_size=10,
                              shuffle=False,
                              num_workers=2,
                              **kwargs,
                             ):
    assert dataset_name in _SEG_DATASETS, f'Dataset not found: {dataset_name}'
    DatasetClass = _SEG_DATASETS[dataset_name]

    dataset = DatasetClass(dataset_type=dataset_type,
                           image_size=image_size,
                           **kwargs,
                          )

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                           )
    return dataloader