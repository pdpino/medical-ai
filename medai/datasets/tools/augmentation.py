import logging
import random
import types
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

from medai.datasets.tools.transforms import (
    pre_transform_masks,
    post_transform_masks,
    RandomRotationMany,
    RandomResizedCropMany,
    RandomAffineMany,
    ColorJitterMany,
    GaussianNoiseMany,
)
from medai.utils.circular_shuffled_list import CircularShuffledList

_SPATIAL_TRANSFORMS = set([
    'crop',
    'translate',
    'rotation',
    'shear',
])
_SHOULD_APPLY_TO_TENSOR = ('noise-gaussian', )
_TRANSFORM_METHOD_NAME = 'transform' # Transform method name in each dataset

AVAILABLE_AUG_MODES = ('single', 'double', 'touch')

LOGGER = logging.getLogger(__name__)


class ImageTransforms:
    """Handle image transformations.

    Augmentation arguments. If None, do not augment with that method; if not None:
        Spatial transforms:
            crop -- minimum relative size of the crop
            translate -- maximum fraction of translation (both vertical and horizontal)
            rotation -- maximum amount of degrees to rotate
            shear -- shear angles (x, y), provided to RandomAffineMany as shear=(-x, x, -y, y)
        Color transforms:
            contrast -- value provided to ColorJitter
            brightness -- value provided to ColorJitter
            noise_gaussian -- Amplifier value to add the gaussian noise
    """
    def __init__(self, image_size,
                 crop=0.8, translate=0.1, rotation=15, contrast=0.8, brightness=0.8,
                 shear=(10, 10), noise_gaussian=0.1):
        self._transform_fns = dict()

        if crop is not None:
            self._transform_fns['crop'] = RandomResizedCropMany(
                image_size,
                scale=(crop, 1),
            )

        if translate is not None:
            self._transform_fns['translate'] = RandomAffineMany(
                degrees=0,
                translate=(translate, translate),
            )

        if shear is not None:
            shear_x, shear_y = shear
            self._transform_fns['shear'] = RandomAffineMany(
                degrees=0,
                shear=(-shear_x, shear_x, -shear_y, shear_y),
            )

        if rotation is not None:
            self._transform_fns['rotation'] = RandomRotationMany(rotation)

        if contrast is not None:
            contrast_down_max = 0.9
            self._transform_fns['contrast-down'] = ColorJitterMany(
                contrast=(contrast_down_max - contrast, contrast_down_max),
            )
            contrast_up_min = 1.1
            self._transform_fns['contrast-up'] = ColorJitterMany(
                contrast=(contrast_up_min, contrast_up_min + contrast),
            )

        if brightness is not None:
            brightness_down_max = 0.9
            self._transform_fns['brightness-down'] = ColorJitterMany(
                brightness=(brightness_down_max - brightness, brightness_down_max),
            )

            brightness_up_min = 1.1
            self._transform_fns['brightness-up'] = ColorJitterMany(
                brightness=(brightness_up_min, brightness_up_min + brightness),
            )

        if noise_gaussian is not None:
            self._transform_fns['noise-gaussian'] = GaussianNoiseMany(amplifier=noise_gaussian)

        self._spatial_transforms = [
            method_name
            for method_name in self._transform_fns
            if method_name in _SPATIAL_TRANSFORMS
        ]

        self._color_transforms = [
            method_name
            for method_name in self._transform_fns
            if method_name not in _SPATIAL_TRANSFORMS
        ]

        # Used if "random-" values provided
        # If none is returned, the original image is used
        self._color_transforms_with_original = list(self._color_transforms) + [None]
        self._spatial_transforms_with_original = list(self._spatial_transforms) + [None]

    def __len__(self):
        return len(self._transform_fns)

    def __iter__(self):
        return iter(self._transform_fns)

    def resolve_aug_method(self, method_name):
        if method_name == 'random-color':
            return random.choice(self._color_transforms_with_original)
        if method_name == 'random-spatial':
            return random.choice(self._spatial_transforms_with_original)

        return method_name

    def get_transform_fn(self, method_name):
        assert isinstance(method_name, str)

        if method_name not in self._transform_fns:
            raise Exception('Internal error: aug method not found')
        return self._transform_fns[method_name]

    @property
    def spatial_transforms(self):
        return self._spatial_transforms

    @property
    def color_transforms(self):
        return self._color_transforms


class Augmentator(Dataset):
    """Augmentates a classification dataset.

    Applies a set of randomized augmentation methods
    """
    def __init__(self, dataset, label=None, force_class=None, times=1, seg_mask=False,
                 dont_shuffle=False, mode='single', **kwargs):
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

        See image-augmentation arguments in the ImageTransforms class
        """
        super().__init__()

        assert mode in AVAILABLE_AUG_MODES, f'Aug-mode not recognized: {mode}'

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
        self._image_transforms = ImageTransforms(dataset.image_size, **kwargs)

        # Create new indices array
        self.mode = mode
        self.n_times = times
        if mode == 'single':
            self.indices = self._build_indices_single(samples_idxs, times)
        elif mode == 'double':
            self.indices = self._build_indices_double(samples_idxs, times)
        elif mode == 'touch':
            self.indices = self._build_indices_touch(samples_idxs, times)

        if not dont_shuffle:
            random.shuffle(self.indices)
        self._did_shuffle = not dont_shuffle

        self.augment_masks = seg_mask
        if not self.augment_masks and dataset.enable_masks:
            LOGGER.error('Passed seg_mask=False and dataset.enable_masks is set to True')

        # Print stats
        stats = {
            'mode': mode,
            'times': times,
            'new-total': f'{len(self.indices):,}',
            'original': f'{len(self.dataset):,}',
            'enable-masks': self.augment_masks,
        }
        stats_str = ' '.join(f'{k}={v}' for k, v in stats.items())
        LOGGER.info('\tAugmenting %s: %s', _samples_info, stats_str)

        self._monkey_patch_transform()

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.indices)

    def _pre_transform_masks_(self, item, fields):
        """Pre-transforms masks in-place in fields."""
        if item.masks.ndim == 2:
            fields['masks'] = pre_transform_masks(item.masks)
        elif item.masks.ndim == 3:
            # HACK: for datasets like VinBig, that return more than one mask,
            # this special case is explictly written
            # FIXME: could the performance be improved? 14 additional transformations
            # are made (all in CPU).
            for i, mask in enumerate(item.masks):
                fields[f'masks-{i}'] = pre_transform_masks(mask)
        else:
            LOGGER.warning('Masks have ndim==%d', item.masks.ndim)

    def _post_transform_masks_(self, item, fields):
        if item.masks.ndim == 2:
            fields['masks'] = post_transform_masks(fields['masks'])
        elif item.masks.ndim == 3:
            n_masks = len(item.masks)

            finished_masks = []
            for i in range(n_masks):
                key = f'masks-{i}'
                finished_masks.append(post_transform_masks(fields[key]))
                del fields[key]

            fields['masks'] = torch.stack(
                finished_masks,
                dim=0,
            ).type(item.masks.type())

    def __getitem__(self, idx):
        inner_idx, aug_spatial, aug_color = self.indices[idx]
        item = self.dataset[inner_idx]

        # If random methods, this resolves to an specific method.
        aug_spatial = self._image_transforms.resolve_aug_method(aug_spatial)
        aug_color = self._image_transforms.resolve_aug_method(aug_color)

        # Grab transforms
        pre_transform, post_transform = self.pre_post_transforms

        # Pre-transform images
        fields = {
            'image': pre_transform(item.image),
        }

        # Pre-transform masks, if needed only
        apply_to_seg_mask = self.augment_masks and aug_spatial in _SPATIAL_TRANSFORMS
        if apply_to_seg_mask:
            self._pre_transform_masks_(item, fields)

        # Augment PIL image with color
        if aug_color is not None and aug_color not in _SHOULD_APPLY_TO_TENSOR:
            aug_fn = self._image_transforms.get_transform_fn(aug_color)
            fields['image'] = aug_fn(fields['image'])

        # Augment PIL image and masks with spatial transforms
        if aug_spatial is not None:
            aug_fn = self._image_transforms.get_transform_fn(aug_spatial)
            fields = aug_fn(fields)

        # Apply post transformation to image
        fields['image'] = post_transform(fields['image'])

        # Apply post-transformation to masks, if needed only
        if apply_to_seg_mask:
            self._post_transform_masks_(item, fields)

        # Augment Tensor image with color, if it should be applied to tensor
        if aug_color in _SHOULD_APPLY_TO_TENSOR:
            aug_fn = self._image_transforms.get_transform_fn(aug_color)
            fields['image'] = aug_fn(fields['image'])

        return item._replace(**fields)

    def _monkey_patch_transform(self):
        """Monkey patches the dataset.transform() function."""
        if not hasattr(self.dataset, _TRANSFORM_METHOD_NAME):
            raise Exception(f'Dataset does not have a method called {_TRANSFORM_METHOD_NAME}')

        tf_instance = getattr(self.dataset, _TRANSFORM_METHOD_NAME)
        if isinstance(tf_instance, types.LambdaType):
            raise Exception('Dataset is already monkey-patched!')

        # Check original transforms
        _transforms = tf_instance.transforms

        assert len(_transforms) >= 2, 'There should be at least two transforms'
        assert isinstance(_transforms[0], transforms.Resize), '1st transform should be Resize'
        assert isinstance(_transforms[1], transforms.ToTensor), '2d transform should be ToTensor'

        pre_transform = _transforms[0] # Resize before
        post_transform = transforms.Compose(_transforms[1:]) # ToTensor after
        self.pre_post_transforms = pre_transform, post_transform

        # Monkey-patch with identity
        setattr(self.dataset, _TRANSFORM_METHOD_NAME, lambda x: x)

    def _build_indices_single(self, samples_idxs, times):
        """Creates an indices array using single-augmentation.

        Single-augmentation: each sample gets augmented once with each aug_method.
        """
        indices = []
        for idx, should_augment in samples_idxs:
            indices.append((idx, None, None)) # Always include the original sample

            if not should_augment:
                continue

            for aug_method in self._image_transforms:
                is_spatial = aug_method in _SPATIAL_TRANSFORMS
                aug_spatial = aug_method if is_spatial else None
                aug_color = aug_method if not is_spatial else None

                for _ in range(times):
                    indices.append((idx, aug_spatial, aug_color))

        return indices

    def _build_indices_double(self, samples_idxs, times):
        """Creates an indices array using double-augmentation.

        Double-augmentation: each sample gets augmented with a spatial and color transform
        at the same time.
        """
        spatial_transforms = CircularShuffledList(self._image_transforms.spatial_transforms)
        color_transforms = CircularShuffledList(self._image_transforms.color_transforms)

        n_transforms = max(len(spatial_transforms), len(color_transforms))

        indices = []
        for idx, should_augment in samples_idxs:
            indices.append((idx, None, None)) # Always include the original sample

            if not should_augment:
                continue

            for _ in range(n_transforms * times):
                aug_method_spatial = next(spatial_transforms)
                aug_method_color = next(color_transforms)
                indices.append((idx, aug_method_spatial, aug_method_color))

        return indices

    def _build_indices_touch(self, samples_idxs, times):
        """Creates an indices array using touch-times-augmentation.

        Touch-augmentation: each sample gets touched with a spatial and color transform
        at the same time. If times == 1, the amount of samples do not increase, but images
        are modified randomly on each iteration.
        """
        indices = []
        for idx, should_augment in samples_idxs:
            if not should_augment:
                indices.append((idx, None, None))
                continue

            for _ in range(times):
                indices.append((idx, 'random-spatial', 'random-color'))

        return indices

    def get_labels_presence_for(self, label):
        """Maps inner dataset indexes to Augmentator indexes.

        This method is overriden to enable OneLabelUnbalancedSampler + Augmentator functionality.
        """

        is_inner_idx_present = dict(self.dataset.get_labels_presence_for(label))

        labels_presence_aug_idxs = [
            (idx, is_inner_idx_present[inner_idx])
            for idx, (inner_idx, _, _) in enumerate(self.indices)
        ]

        return labels_presence_aug_idxs

    def plot_augmented_samples(self, sample_idx, n_masks=1, title_fontsize=15):
        """Utility function to plot augmented samples.

        The instance must have been created with dont_shuffle=True,
        so the samples are ordered!!

        NOTE: For datasets like VinBig, that item.masks returned has an additional dimension,
        (i.e. the disease dimension first: n_diseases, height, width), n_masks must be
        explictly provided to account for this and plot all the masks. In the future, this
        could be inferred from the dataset.
        (At the moment, it can be infered from the item.masks.ndim, but the amount of masks
        must be determined before accessing an item, to set n_rows properly).
        """
        assert not self._did_shuffle, 'The augmented dataset was shuffled!'

        def _prettify_method_name(methods):
            if not isinstance(methods, (tuple, list)):
                methods = (methods,)

            _mapping = {
                'rotation': 'rot',
                'brightness': 'b',
                'contrast': 'c',
                'noise-gaussian': 'gauss',
                'translate': 'trans',
                'random-color': 'rnd-c',
                'random-spatial': 'rnd-s',
            }
            pretty_methods = []
            for method in methods:
                if method is None:
                    continue

                for k, v in _mapping.items():
                    method = method.replace(k, v)
                pretty_methods.append(method)

            return ','.join(pretty_methods)


        should_plot_masks = self.augment_masks and n_masks >= 1
        should_plot_original = 1

        # Amounts
        if self.mode == 'single':
            n_aug_methods = len(self._image_transforms)
        elif self.mode == 'double':
            n_color_methods = len(self._image_transforms.color_transforms)
            n_spatial_methods = len(self._image_transforms.spatial_transforms)
            n_aug_methods = max(n_color_methods, n_spatial_methods)
        elif self.mode == 'touch':
            n_aug_methods = 1
            should_plot_original = 0

        n_aug_methods *= self.n_times

        n_cols = n_aug_methods + should_plot_original
        n_rows = 1 + n_masks if should_plot_masks else 1

        # Augmented sample idx
        base_idx = sample_idx * (n_aug_methods + should_plot_original)

        plt.figure(figsize=(n_cols*5, n_rows*5))

        def _plot_mask(mask, plot_index):
            plt.subplot(n_rows, n_cols, plot_index)
            plt.imshow(mask)
            plt.axis('off')

        def _plot_item_masks(item, base_plot_index):
            if item.masks.ndim == 2:
                _plot_mask(item.masks, base_plot_index)
            elif item.masks.ndim == 3:
                n_actual_masks = len(item.masks)
                if n_actual_masks != n_masks:
                    raise Exception(f'Received n_masks={n_masks}, should be {n_actual_masks}')
                for i, mask in enumerate(item.masks):
                    _plot_mask(mask, base_plot_index + i*n_cols)

        if should_plot_original:
            item = self[base_idx]
            plt.subplot(n_rows, n_cols, 1)
            plt.title('original', fontsize=title_fontsize)
            plt.imshow(item.image[0], cmap='gray')
            plt.axis('off')

            if should_plot_masks:
                _plot_item_masks(item, n_cols + 1)

        for i in range(n_aug_methods):
            # Move idx to the right
            augmented_idx = base_idx + should_plot_original + i

            # Get method(s)
            method = self.indices[augmented_idx][1:]

            item = self[augmented_idx]
            plt.subplot(n_rows, n_cols, i + 1 + should_plot_original)
            plt.title(_prettify_method_name(method), fontsize=title_fontsize)
            plt.imshow(item.image[0], cmap='gray')
            plt.axis('off')

            if should_plot_masks:
                _plot_item_masks(item, n_cols + i + 1 + should_plot_original)
