from torch.utils.data import DataLoader

from mrg.datasets.cxr14 import CXR14Dataset, CXR14UnbalancedSampler
from mrg.datasets.covid_kaggle import CovidKaggleDataset

_DATASET_DEF = {
  'cxr14': CXR14Dataset,
  'covid-kaggle': CovidKaggleDataset,
}

AVAILABLE_DATASETS = list(_DATASET_DEF)

def prepare_data_classification(dataset_name='cxr14', dataset_type='train', labels=None,
                                max_samples=None,
                                oversample=False, max_os=None,
                                batch_size=10, shuffle=False):
    print(f'Loading {dataset_type} dataset...')

    assert dataset_name in _DATASET_DEF, f'Dataset not found: {dataset_name}'
    DatasetClass = _DATASET_DEF[dataset_name]

    dataset = DatasetClass(dataset_type=dataset_type,
                           labels=labels,
                           max_samples=max_samples)

    if oversample:
        if dataset_name != 'cxr14':
            # FIXME: only works with CXR14 dataset
            raise Exception('Oversampler only works with CXR14')
        sampler = CXR14UnbalancedSampler(dataset, max_os)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader

