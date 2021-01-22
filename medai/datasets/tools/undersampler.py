import random
import logging
from torch.utils.data import Sampler

LOGGER = logging.getLogger(__name__)

class OneLabelUnderSampler(Sampler):
    def __init__(self, dataset, label=0):
        """Undersample a majority class for a given label.

        Args:
            dataset -- torch.utils.data.Dataset that implements `get_labels_presence_for(label)`
            label -- str (label name) or int (index)
        """
        super().__init__(dataset)

        total_samples = len(dataset)

        # Labels for each sample
        labels_presence_by_idx = dataset.get_labels_presence_for(label)

        # Get indices for each class
        positives = [idx for idx, presence in labels_presence_by_idx if presence]
        negatives = [idx for idx, presence in labels_presence_by_idx if not presence]
        n_positives = len(positives)
        n_negatives = len(negatives)

        # Sample majority class randomly
        if n_positives < n_negatives:
            should_normally_sample = positives
            should_undersample = negatives
            undersampled_class = 'negative'
        else:
            should_normally_sample = negatives
            should_undersample = positives
            undersampled_class = 'positive'

        normally_sampled = should_normally_sample
        undersampled = random.sample(should_undersample, len(should_normally_sample))

        self.resampled_indexes = normally_sampled + undersampled

        random.shuffle(self.resampled_indexes)

        # Print info
        label_name = dataset.labels[label] if isinstance(label, int) else label
        stats = {
            'positives': n_positives,
            'negatives': n_negatives,
            'new-total': len(self.resampled_indexes),
            'original': total_samples,
        }
        stats_str = ' '.join(f'{k}={v}' for k, v in stats.items())
        LOGGER.info(
            '\tUndersampling %s (%s): %s',
            label_name,
            undersampled_class,
            stats_str,
        )

    def __len__(self):
        return len(self.resampled_indexes)

    def __iter__(self):
        return iter(self.resampled_indexes)
