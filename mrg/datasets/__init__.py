from torch.utils.data import DataLoader

from mrg.datasets.cxr14 import CXR14Dataset
from mrg.datasets.covid_kaggle import CovidKaggleDataset
from mrg.datasets.sampler import OneLabelUnbalancedSampler

_DATASET_DEF = {
  'cxr14': CXR14Dataset,
  'covid-kaggle': CovidKaggleDataset,
}

AVAILABLE_CLASSIFICATION_DATASETS = list(_DATASET_DEF)

def prepare_data_classification(dataset_name='cxr14', dataset_type='train', labels=None,
                                max_samples=None,
                                oversample=False, oversample_label=0, oversample_max_ratio=None,
                                batch_size=10, shuffle=False):
    print(f'Loading {dataset_type} dataset...')

    assert dataset_name in _DATASET_DEF, f'Dataset not found: {dataset_name}'
    DatasetClass = _DATASET_DEF[dataset_name]

    dataset = DatasetClass(dataset_type=dataset_type,
                           labels=labels,
                           max_samples=max_samples)

    if oversample:
        sampler = OneLabelUnbalancedSampler(dataset,
                                            label=oversample_label,
                                            max_ratio=oversample_max_ratio)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader

