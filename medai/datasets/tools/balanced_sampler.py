import logging
from torch.utils.data import Sampler

from medai.utils.circular_shuffled_list import CircularShuffledList

LOGGER = logging.getLogger(__name__)

class MultilabelBalancedSampler(Sampler):
    def __init__(self, dataset, include_nf=True, cap_max_len=20000):
        """Samples items balancing by label.

        Process:
            1. Put positive samples from each label into a (circular-shuffled)-bucket
            2. `max_len` is the max size of the buckets
            3. Sample one sample from each bucket sequentially
            4. Repeat that `max_len` times, until all the buckets are exhausted

        The step 3 ensures that each batch will be (roughly) balanced,
        with one sample from each label.
        The step 4 ensures all samples are used (though if `cap_max_len` is given, this will
        not hold for the most represented labels (such as 'No Finding')).

        FIXME: for now, is only tested with multilabel=True datasets,
        though it should work with multilabel=False as well.

        Args:
            dataset -- torch.utils.data.Dataset that implements `get_labels_presence_for()`, and
                `get_presence_for_no_finding` if the 'No Finding' disease is not in their labels.
            include_nf -- If present and the 'No Finding' disease is not in their labels, include
                it anyway.
            cap_max_len -- Maximum for the max_len value
        """
        super().__init__(dataset)

        LABELS = list(dataset.labels)

        presence_by_label = [
            CircularShuffledList(
                index
                for index, present in dataset.get_labels_presence_for(label)
                if present
            )
            for label in LABELS
        ]

        if 'No Finding' not in LABELS and include_nf:
            presence_by_label.append(CircularShuffledList(
                index
                for index, present in dataset.get_presence_for_no_finding()
                if present
            ))

        n_samples_by_label = [len(l) for l in presence_by_label]
        max_len = max(n_samples_by_label)
        if cap_max_len is not None:
            max_len = min(max_len, cap_max_len)

        self.resampled_indexes = []
        presence_by_label = CircularShuffledList(presence_by_label)
        for _ in range(max_len):
            for _ in range(len(presence_by_label)):
                samples_with_label_present = presence_by_label.get_next()
                selected_sample = samples_with_label_present.get_next()

                if selected_sample is not None:
                    self.resampled_indexes.append(selected_sample)

        # Print info
        stats = {
            'samples-by-label': n_samples_by_label,
            'cap-max-samples': cap_max_len,
            'new-total': len(self.resampled_indexes),
            'original': len(dataset),
            'times-larger': len(self.resampled_indexes) // len(dataset),
        }
        stats_str = ' '.join(f"{k}={v}" for k, v in stats.items())
        LOGGER.info('\tMultilabel balanced-sampling: %s', stats_str)

    def __len__(self):
        return len(self.resampled_indexes)

    def __iter__(self):
        return iter(self.resampled_indexes)
