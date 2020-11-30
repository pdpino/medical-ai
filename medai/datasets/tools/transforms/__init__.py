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
    elif isinstance(images, (list, tuple)):
        return [
            transform_fn(image) for image in images
        ]
    elif isinstance(images, Image.Image):
        return transform_fn(images)
    raise Exception(f'images type not supported: {type(images)}')


def get_first(images):
    if isinstance(images, Image.Image):
        return images
    elif isinstance(images, (list, tuple)):
        return images[0]
    elif isinstance(images, dict):
        key = next(iter(images))
        return images[key]
    raise Exception(f'images type not supported: {type(images)}')


class RandomRotationMany(transforms.RandomRotation):
    def __call__(self, images):
        angle = self.get_params(self.degrees)
        transform_fn = lambda image: F.rotate(
            image, angle, self.resample, self.expand, self.center, self.fill)

        return apply_to_many(transform_fn, images)


class RandomResizedCropMany(transforms.RandomResizedCrop):
    def __call__(self, images):
        first_image = get_first(images)
        i, j, h, w = self.get_params(first_image, self.scale, self.ratio)
        transform_fn = lambda image: F.resized_crop(
            image, i, j, h, w, self.size, self.interpolation)

        return apply_to_many(transform_fn, images)


class RandomAffineMany(transforms.RandomAffine):
    def __call__(self, images):
        first_image = get_first(images)
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, first_image.size)

        transform_fn = lambda image: F.affine(
            image, *ret, resample=self.resample, fillcolor=self.fillcolor)

        return apply_to_many(transform_fn, images)


class ColorJitterMany(transforms.ColorJitter):
    def __call__(self, images):
        transform_fn = self.get_params(self.brightness, self.contrast,
                                       self.saturation, self.hue)
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

    Args:
        masks_pil -- PIL image of shape (height, width)
    Returns:
        Tensor of shape (height, width), of type long
    """
    masks = torch.tensor(np.array(masks_pil)) # shape: height, width
    # masks = masks.unsqueeze(0) # shape: 1, height, width
    masks = masks.long()
    return masks