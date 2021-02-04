from functools import reduce

import torch
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image


def apply_to_many(transform_fn, images):
    if isinstance(images, dict):
        return {
            name: transform_fn(image)
            for name, image in images.items()
        }
    if isinstance(images, (list, tuple)):
        return [
            transform_fn(image) for image in images
        ]
    if isinstance(images, (Image.Image, torch.Tensor)):
        return transform_fn(images)
    raise Exception(f'images type not supported: {type(images)}')


def get_first(images):
    if isinstance(images, (Image.Image, torch.Tensor)):
        return images
    if isinstance(images, (list, tuple)):
        return images[0]
    if isinstance(images, dict):
        key = next(iter(images))
        return images[key]
    raise Exception(f'images type not supported: {type(images)}')


### NOTE:
### Transform functions implemented for many images at the same time are implemented
### for torch version 1.7.1, copying the implementation of the base-class' forward() method.
### If the forward() methods change in a future pytorch/torchvision version,
### the many-re-implementation may produce a wrong output or fail.


class RandomRotationMany(transforms.RandomRotation):
    def forward(self, images): # pylint: disable=arguments-differ
        angle = self.get_params(self.degrees)
        transform_fn = lambda image: F.rotate(
            image, angle, self.resample, self.expand, self.center, self.fill)

        return apply_to_many(transform_fn, images)


class RandomResizedCropMany(transforms.RandomResizedCrop):
    def forward(self, images): # pylint: disable=arguments-differ
        first_image = get_first(images)
        i, j, h, w = self.get_params(first_image, self.scale, self.ratio)
        transform_fn = lambda image: F.resized_crop(
            image, i, j, h, w, self.size, self.interpolation)

        return apply_to_many(transform_fn, images)


class RandomAffineMany(transforms.RandomAffine):
    def forward(self, images): # pylint: disable=arguments-differ
        first_image = get_first(images)
        ret = self.get_params(
            self.degrees, self.translate, self.scale, self.shear, first_image.size,
        )

        transform_fn = lambda image: F.affine(
            image, *ret, resample=self.resample, fillcolor=self.fillcolor)

        return apply_to_many(transform_fn, images)


class ColorJitterMany(transforms.ColorJitter):
    def _build_transform_function(self):
        """Builds a function that transforms an image.

        Copied from ColorJitter.forward() method."""
        fn_idx = torch.randperm(4)

        transformations_to_apply = []

        for fn_id in fn_idx:
            # pylint: disable=not-callable
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                transformations_to_apply.append((F.adjust_brightness, brightness_factor))

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                transformations_to_apply.append((F.adjust_contrast, contrast_factor))

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                transformations_to_apply.append((F.adjust_saturation, saturation_factor))

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                transformations_to_apply.append((F.adjust_hue, hue_factor))

        def _apply_transform(value, tup):
            transform_fn, arg = tup
            return transform_fn(value, arg)
        transform = lambda img: reduce(_apply_transform, transformations_to_apply, img)

        return transform

    def __call__(self, images):
        transform_fn = self._build_transform_function()

        return apply_to_many(transform_fn, images)


class GaussianNoiseMany:
    """Adds gaussian noise to images.

    Assumes images are tensors of shape (n_channels, height, width)
    """
    def __init__(self, amplifier):
        self.amplifier = amplifier

    def __call__(self, images):
        image = get_first(images)
        unused_n_channels, height, width = image.size()
        noise = torch.randn(height, width, device=image.device) * self.amplifier

        transform_fn = lambda image: image + noise

        return apply_to_many(transform_fn, images)


class WIP_TransformMany:
    """WIP: Class not implemented.

    Transform classes do not follow same pattern for get_params and __call__ methods
    (see above implementations, <TransformClass>Many cases).
    """
    def __init__(self, tf_class, tf_fn, get_params={}, extra_params={}):
        self.tf_class = tf_class
        self.tf_fn = tf_fn

        self.get_params = get_params
        self.extra_params = extra_params

    def __call__(self, imgs, first_key='image'):
        """Transforms a dict of images with the same transformation.

        Args
            imgs -- dict of (keys=image_name, values=pil_image)
            first -- key to use as first key of the dict
        """
        raise NotImplementedError
        # pylint: disable=unreachable
        # fn_params = self.tf_class.get_params(imgs[first_key], **self.get_params)
        fn_params = self.tf_class.get_params(**self.get_params)

        tf_imgs = {
            img_name: self.tf_fn(img, *fn_params, **self.extra_params)
            for img_name, img in imgs.items()
        }

        return tf_imgs



def pre_transform_masks(masks):
    """Steps to perform before applying a transform to a segmentation mask.

    Args:
        masks -- tensor of shape (height, width), with values between [0, 255]
    Returns:
        PIL Image of size (height, width)
    """
    masks = masks.to(torch.uint8)
    masks = F.to_pil_image(masks, 'L')
    return masks

def post_transform_masks(masks_pil):
    """Steps to perform after applying a transform to a segmentation mask.

    Note: the pil -> numpy -> tensor transormation is done manually,
    and not using the `transforms.ToTensor()` class, to avoid range
    modifications (i.e. from 0-1, 0-255, etc).

    Args:
        masks_pil -- PIL image of shape (height, width)
    Returns:
        Tensor of shape (height, width), of type long
    """
    masks = torch.from_numpy(np.array(masks_pil)) # shape: height, width
    # masks = masks.unsqueeze(0) # shape: 1, height, width
    masks = masks.long()
    return masks
