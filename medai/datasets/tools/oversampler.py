import random
from torch.utils.data import Sampler

class OneLabelOverSampler(Sampler):
    def __init__(self, dataset, label=0, ratio=None, force_class=None, max_ratio=None):
        """Oversamples samples that are less represented, considering a specific label.
        
        Used to oversample one under-represented class.
        Tries to oversample the given `label`, by a factor that implies that
            n_positive_cases == n_negative_cases, or `max_ratio`.
        The dataset must implement the method `dataset.get_labels_presence_for(label)`,
            see CXR14Dataset or CovidKaggleDataset for multilabel or non-multilabel examples,
            respectively.

        Args:
            dataset -- torch.utils.data.Dataset that implements `get_labels_presence_for(label)`
            label -- str (label name) or int (index)
            ratio -- int. Specifies amount of times to oversample. If not provided,
                calculates ratio to level negative and positive samples.
            force_class -- 0, 1 or None
                If not None, forcedly oversample samples with class provided (1=positive, 0=negative)
            max_ratio -- float indicating max oversampling ratio allowed
        """
        total_samples = len(dataset)

        # Labels for each sample
        labels_presence_by_idx = dataset.get_labels_presence_for(label)

        # Compute oversampling ratio
        positives = sum(presence for idx, presence in labels_presence_by_idx)
        negatives = total_samples - positives

        # Choose class to sample (0 or 1, absence or presence)
        if force_class is None:
            # Oversample the minority class by default
            OVERSAMPLE_CLASS = 1 if negatives > positives else 0
        else:
            # Unless user wants to force oversampling a specific class
            assert force_class in (0, 1), f'force_class must be 0 or 1, got: {force_class}'
            OVERSAMPLE_CLASS = force_class

        UNDERSAMPLE_CLASS = 1 - OVERSAMPLE_CLASS

        if ratio is None:
            if OVERSAMPLE_CLASS == 1:
                ratio = negatives // positives if positives > 0 else 1
            else: # OVERSAMPLE_CLASS == 0
                ratio = positives // negatives if negatives > 0 else 1

        # If ratio is still 0, means positive ~~ negatives
        # --> don't oversample --> set ratio = 1
        if ratio < 1:
            ratio = 1

        # Set a maximum ratio for oversampling
        # note that it only affects ratio > 1 (i.e. oversampling positive samples)
        if max_ratio is not None:
            ratio = min(ratio, max_ratio)
        
        # Resample indexes
        self.resampled_indexes = []
        
        for idx, presence in labels_presence_by_idx:
            if presence == UNDERSAMPLE_CLASS:
                self.resampled_indexes.append(idx)
            elif presence == OVERSAMPLE_CLASS:
                for _ in range(ratio):
                    self.resampled_indexes.append(idx)

        # Shuffle to avoid having oversampled samples together
        random.shuffle(self.resampled_indexes)
        
        # Print info    
        label_name = dataset.labels[label] if isinstance(label, int) else label
        stats = {
            'os-class': OVERSAMPLE_CLASS,
            'ratio': ratio,
            'positives': positives,
            'negatives': negatives,
            'new-total': len(self.resampled_indexes),
            'original': total_samples,
        }
        print(f'\tOversampling {label_name}:', ' '.join(f"{k}={v}" for k, v in stats.items()))

    
    def __len__(self):
        return len(self.resampled_indexes)
    
    def __iter__(self):
        return iter(self.resampled_indexes)
        