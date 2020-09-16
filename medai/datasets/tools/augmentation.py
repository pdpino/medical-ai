import random
from torch.utils.data import Dataset
from torchvision import transforms

class Augmentator(Dataset):
    """Augmentates a classification dataset.
    
    Dataset class must have:
        - method `get_labels_presence_for(label)`, see OneLabelUnbalancedSampler
        - attribute `image_size: tuple`, with (height, width) format
    """
    def __init__(self, dataset, label=None, crop=0.8, translate=0.1, rotation=15,
                 contrast=0.5, brightness=0.5):
        self.dataset = dataset

        # Define which samples to augment
        if label is None:
            # Augment samples from all labels
            iterator = [(idx, True) for idx in range(len(self.dataset))]
            _augmented_samples = 'all samples'
        else:
            # Only augment samples from a specific label
            label_name = dataset.labels[label] if isinstance(label, int) else label
            iterator = dataset.get_labels_presence_for(label)
            _augmented_samples = f'samples only from label {label_name}'


        # Define augmentation methods
        self._aug_fns = dict()
        if crop is not None:
            self._aug_fns['crop'] = transforms.RandomResizedCrop(
                dataset.image_size,
                scale=(crop, 1),
            )

        if translate is not None:
            self._aug_fns['translate'] = transforms.RandomAffine(
                0, # degrees
                translate=(translate, translate),
            )

        if rotation is not None:
            self._aug_fns['rotation'] = transforms.RandomRotation(rotation)

        if contrast is not None:
            self._aug_fns['contrast'] = transforms.ColorJitter(contrast=contrast)

        if brightness is not None:
            self._aug_fns['brightness'] = transforms.ColorJitter(brightness=brightness)

        # dataset returns tensor
        # tensor --> pil --> transform --> tensor
        # FIXME: more efficient way?
        self._to_pil = transforms.ToPILImage()
        self._to_tensor = transforms.ToTensor()

        # Create new indices array
        self.indices = []
        for idx, presence in iterator:
            self.indices.append((idx, None))

            if not presence:
                continue

            for aug_method in self._aug_fns.keys():
                self.indices.append((idx, aug_method))

        random.shuffle(self.indices)

        # Print stats
        stats = {
            'new-total': len(self.indices),
            'original': len(self.dataset),
        }
        stats_str = ' '.join(f'{k}={v}' for k, v in stats.items())
        print(f'\tAugmenting {_augmented_samples}: ', stats_str)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        inner_idx, aug_method = self.indices[idx]

        item = self.dataset[inner_idx]
        if aug_method is None:
            return item
        aug_fn = self._aug_fns[aug_method]

        image = self._to_pil(item[0])
        image = self._to_tensor(aug_fn(image))

        return item._replace(image=image)


    def get_labels_presence_for(self, label):
        """Maps inner dataset indexes to Augmentator indexes.
        
        This method is overriden to enable OneLabelUnbalancedSampler + Augmentator functionality.
        """
        
        is_inner_idx_present = {
            idx: presence
            for idx, presence in self.dataset.get_labels_presence_for(label)
        }

        labels_presence_aug_idxs = [
            (idx, is_inner_idx_present[inner_idx])
            for idx, (inner_idx, _) in enumerate(self.indices)
        ]

        return labels_presence_aug_idxs
        
