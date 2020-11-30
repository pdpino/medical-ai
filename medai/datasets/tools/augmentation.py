import random
import types
from torch.utils.data import Dataset
from torchvision import transforms

from medai.datasets.tools.transforms import (
    pre_transform_masks,
    post_transform_masks,
    RandomRotationMany,
    RandomResizedCropMany,
    RandomAffineMany,
    ColorJitterMany,
)

_SPATIAL_TRANSFORMS = set([
    'crop',
    'translate',
    'rotation',
    'shear',
    ])
_TRANSFORM_METHOD_NAME = 'transform' # Transform method name in each dataset

class Augmentator(Dataset):
    """Augmentates a classification dataset.

    Applies a set of randomized augmentation methods
    """
    def __init__(self, dataset, label=None, force_class=None, times=1, seg_mask=False,
                 dont_shuffle=False,
                 crop=0.8, translate=0.1, rotation=15, contrast=0.5, brightness=0.5,
                 shear=(10, 10),
                 ):
        """Init class.

        Args
            dataset -- Dataset to augment. Class must implement:
                - method `get_labels_presence_for(label)`, see OneLabelUnbalancedSampler
                - attribute `image_size: tuple`, with (height, width) format
            label -- If provided, only augment positive samples from that label
            force_class -- If provided, force using positive/negative samples from the `label`
                selected
            times -- number of times to augment the same sample with the same augment method
            seg_mask -- If present also augment `masks` field from the `BatchItem`s
                (by default, only augment the `image` field)
            dont_shuffle -- do not shuffle augmented samples (used for debugging)

        Augmentation arguments. If None, do not augment with that method; if not None:
            crop -- minimum relative size of the crop
            translate -- maximum fraction of translation (both vertical and horizontal)
            rotation -- maximum amount of degrees to rotate
            contrast -- value provided to ColorJitter
            brightness -- value provided to ColorJitter
            shear -- shear angles (x, y), provided to RandomAffineMany as shear=(-x, x, -y, y)
        """
        self.dataset = dataset

        # Define which samples to augment
        if label is None:
            # Augment samples from all labels
            samples_idxs = [(idx, True) for idx in range(len(self.dataset))]
            _samples_info = 'all samples'
        else:
            # Only augment samples from a specific label
            label_name = dataset.labels[label] if isinstance(label, int) else label

            # ...and with an specific class
            target_class = force_class if force_class is not None else 1

            samples_idxs = [
                (idx, presence == target_class)
                for idx, presence in dataset.get_labels_presence_for(label)
            ]
            _samples_info = f'samples only from label {label_name}, with class={target_class}'


        # Define augmentation methods
        self._aug_fns = dict()
        if crop is not None:
            self._aug_fns['crop'] = RandomResizedCropMany(
                dataset.image_size,
                scale=(crop, 1),
            )

        if translate is not None:
            self._aug_fns['translate'] = RandomAffineMany(
                degrees=0,
                translate=(translate, translate),
            )

        if shear is not None:
            shear_x, shear_y = shear
            self._aug_fns['shear'] = RandomAffineMany(
                degrees=0,
                shear=(-shear_x, shear_x, -shear_y, shear_y),
            )

        if rotation is not None:
            self._aug_fns['rotation'] = RandomRotationMany(rotation)

        if contrast is not None:
            self._aug_fns['contrast'] = ColorJitterMany(contrast=contrast)

        if brightness is not None:
            self._aug_fns['brightness'] = ColorJitterMany(brightness=brightness)

        # Create new indices array
        self.indices = []
        for idx, should_augment in samples_idxs:
            self.indices.append((idx, None)) # Always include the original sample

            if not should_augment:
                continue

            for aug_method in self._aug_fns.keys():
                for _ in range(times):
                    self.indices.append((idx, aug_method))

        if not dont_shuffle:
            random.shuffle(self.indices)

        self.augment_masks = seg_mask

        # Print stats
        stats = {
            'times': times,
            'new-total': len(self.indices),
            'original': len(self.dataset),
        }
        stats_str = ' '.join(f'{k}={v}' for k, v in stats.items())
        print(f'\tAugmenting {_samples_info}: ', stats_str)

        self.monkey_patch_transform()

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        inner_idx, aug_method = self.indices[idx]

        item = self.dataset[inner_idx]

        # Prepare PIL images
        fields = {
            'image': self.pre_transform(item.image),
        }

        apply_to_seg_mask = self.augment_masks and aug_method in _SPATIAL_TRANSFORMS
        if apply_to_seg_mask:
            fields['masks'] = pre_transform_masks(item.masks)

        # Apply transformation
        if aug_method is not None:
            aug_instance = self._aug_fns[aug_method]
            fields = aug_instance(fields)

        # Apply post transformation
        fields['image'] = self.post_transform(fields['image'])

        if apply_to_seg_mask:
            fields['masks'] = post_transform_masks(fields['masks'])

        return item._replace(**fields)


    def monkey_patch_transform(self):
        """Monkey patches the dataset.transform() function."""
        if not hasattr(self.dataset, _TRANSFORM_METHOD_NAME):
            raise Exception(f'Dataset does not have a method called {_TRANSFORM_METHOD_NAME}')

        tf_instance = getattr(self.dataset, _TRANSFORM_METHOD_NAME)
        if isinstance(tf_instance, types.LambdaType):
            raise Exception('Dataset is already monkey-patched!')

        # Check original transforms
        _transforms = tf_instance.transforms

        assert len(_transforms) >= 2, 'There should be at least two transforms'
        assert isinstance(_transforms[0], transforms.Resize), 'First transform should be Resize'
        assert isinstance(_transforms[1], transforms.ToTensor), 'Second transform should be ToTensor'

        # Break in two steps
        self.pre_transform = _transforms[0]
        self.post_transform = transforms.Compose(_transforms[1:])

        # Set as identity
        setattr(self.dataset, _TRANSFORM_METHOD_NAME, lambda x: x)


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

