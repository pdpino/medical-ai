import numpy as np
from torch.utils.data import DataLoader, Subset

from mrg.datasets.cxr14 import CXR14Dataset
from mrg.datasets.covid_kaggle import CovidKaggleDataset
from mrg.datasets.covid_x import CovidXDataset
from mrg.datasets.covid_actual import CovidActualDataset
from mrg.datasets.covid_fig1 import CovidFig1Dataset
from mrg.datasets.covid_uc import CovidUCDataset

from mrg.datasets.tools.oversampler import OneLabelOverSampler
from mrg.datasets.tools.undersampler import OneLabelUnderSampler
from mrg.datasets.tools.augmentation import Augmentator

from mrg.utils.nlp import count_sentences

_DATASET_DEF = {
  'cxr14': CXR14Dataset,
  'covid-kaggle': CovidKaggleDataset,
  'covid-x': CovidXDataset,
  'covid-actual': CovidActualDataset,
  'covid-fig1': CovidFig1Dataset,
  'covid-uc': CovidUCDataset,
}

AVAILABLE_CLASSIFICATION_DATASETS = list(_DATASET_DEF)

def prepare_data_classification(dataset_name='cxr14', dataset_type='train', labels=None,
                                max_samples=None, image_size=(512, 512),
                                augment=False, augment_label=None, augment_kwargs={},
                                oversample=False, oversample_label=0, oversample_max_ratio=None,
                                undersample=False, undersample_label=0,
                                batch_size=10, shuffle=False,
                                **kwargs,
                                ):
    print(f'Loading {dataset_name}/{dataset_type} dataset...')

    assert dataset_name in _DATASET_DEF, f'Dataset not found: {dataset_name}'
    DatasetClass = _DATASET_DEF[dataset_name]

    if 'frontal_only' in kwargs and dataset_name != 'covid-uc':
        print('\tWarning: frontal-only option is only implemented in covid-uc')

    dataset = DatasetClass(dataset_type=dataset_type,
                           labels=labels,
                           image_size=image_size,
                           max_samples=max_samples,
                           **kwargs,
                           )

    if len(dataset) == 0:
        return None

    if augment:
        dataset = Augmentator(dataset, label=augment_label, **augment_kwargs)

    if oversample:
        sampler = OneLabelOverSampler(dataset,
                                            label=oversample_label,
                                            max_ratio=oversample_max_ratio)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    elif undersample:
        sampler = OneLabelUnderSampler(dataset, label=undersample_label)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
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

