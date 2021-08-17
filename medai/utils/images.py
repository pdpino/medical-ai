import os
import logging
import numbers
from tqdm.auto import tqdm

import numpy as np
import torch
from torch.nn.functional import interpolate
from torchvision import transforms
from ignite.utils import to_onehot
from PIL import Image

from medai.datasets.common import BatchItem
from medai.utils import tensor_to_range01


LOGGER = logging.getLogger(__name__)

class ImageFolderIterator:
    def __init__(self, folder, image_names, image_format='RGB'):
        self.folder = folder
        self.image_names = image_names

        self.image_format = image_format

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_names)

    def __iter__(self):
        for image_name in self.image_names:
            fpath = os.path.join(self.folder, image_name)
            image = Image.open(fpath)

            if self.image_format:
                image = image.convert(self.image_format)

            image = self.transform(image)

            yield image


def compute_mean_std(image_iterator, n_channels=3, show=False, threads=1):
    """Computes mean and std of a dataset.

    Args:
        image_iterator -- should yield one image at the time; each image is
            a tensor of shape (n_channels, height, width)

    Returns:
      Channel wise mean, std (i.e. tensors of shape n_channels)
    """
    torch.set_num_threads(threads)

    mean = torch.zeros(n_channels)
    std = torch.zeros(n_channels)

    n_samples = 0

    if show:
        image_iterator = tqdm(image_iterator)

    for image in image_iterator:
        image_flatten = image.view(image.size(0), -1)

        mean += image_flatten.mean(1)
        std += image_flatten.std(1)
        # shapes: n_channels

        n_samples += 1

    mean /= n_samples
    std /= n_samples

    return mean, std


class NormalizeBySample:
    """Normalizes images with their mean and std."""
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon

    def __call__(self, image):
        n_channels, unused_height, unused_width = image.size()
        sample = image.view(n_channels, -1)

        sample_mean = sample.mean(-1).unsqueeze(-1).unsqueeze(-1)
        sample_std = sample.std(-1).unsqueeze(-1).unsqueeze(-1)
        # shape: n_channels, height, width

        return (image - sample_mean) / (sample_std + self.epsilon)


class ScaleValues:
    """Scale values to a [-N, N] range."""
    def __init__(self, target=1024):
        self.target = target

    def __call__(self, image):
        n_channels, unused_height, unused_width = image.size()
        sample = image.view(n_channels, -1)

        sample_max = sample.max()
        sample_range = (sample_max - sample.min())

        slope = self.target * 2 / sample_range
        intersect = self.target - 2 * self.target * sample_max.true_divide(sample_range)

        return image * slope + intersect


class GrayTo3Channels:
    def __call__(self, img):
        """Converts image to 3 channels.
        If image has 3 channels, returns unchanged
        If image has 1 channel, repeats three times as RGB
        img size: n_channels, height, width
        """
        assert isinstance(img, torch.Tensor)

        if img.ndim == 2:
            img = img.unsqueeze(0)

        assert img.ndim == 3

        n_channels = img.size(0)

        if n_channels == 3:
            return img

        assert n_channels == 1, f'Received invalid n_channels: {n_channels}'

        return img.repeat(3, 1, 1)
        # shape: 3, height, width


class PILToTensorAndRange01:
    def __call__(self, pic):
        """Transform PIL image to tensor and to range 0-1.
        The class transforms.ToTensor() transforms 8bit images to 0-1 range,
        but not 16bit images.
        This class transform images from any type to tensor and to 0-1 range.
        """
        # Copied from torchvision.transforms
        arr = np.asarray(pic).astype(np.float32)
        tensor = torch.as_tensor(arr, dtype=torch.float)
        tensor = tensor.view(pic.size[1], pic.size[0], len(pic.getbands()))
        tensor = tensor.permute((2, 0, 1))
        # shape: n_channels, height, width

        # To range 0-1
        tensor -= tensor.min()
        tensor /= tensor.max()
        return tensor


def get_default_image_transform(image_size=(512, 512),
                                norm_by_sample=True,
                                mean=0,
                                std=1,
                                xrv_norm=False,
                                crop_center=None,
                                bit16=False,
                                ):
    def _to_channel_list(value):
        if isinstance(value, (list, tuple)):
            return value
        return [value]

    tfs = [
        transforms.Resize(image_size),
    ]

    if crop_center is not None:
        tfs.append(transforms.CenterCrop(crop_center))

    if bit16:
        tfs.extend([
            PILToTensorAndRange01(),
            GrayTo3Channels(),
        ])
    else:
        tfs.append(transforms.ToTensor())

    if norm_by_sample:
        tfs.append(NormalizeBySample())
    else:
        mean = _to_channel_list(mean)
        std = _to_channel_list(std)
        tfs.append(transforms.Normalize(mean, std))

    if xrv_norm:
        tfs.append(ScaleValues(target=1024))

    return transforms.Compose(tfs)


def bbox_coordinates_to_map(bboxes, valid, image_size):
    """Transform bbox (x,y,w,h) pairs to binary maps.

    Args:
      bboxes - tensor of shape (batch_size, n_diseases, 4)
      valid - tensor of shape (batch_size, n_diseases)
    Returns:
      tensor of shape (batch_size, n_diseases, height, width)
    """
    batch_size, n_labels = valid.size()
    bboxes_map = torch.zeros(batch_size, n_labels, *image_size, device=bboxes.device)

    for i_batch in range(batch_size):
        for j_label in range(n_labels):
            if valid[i_batch, j_label].item() == 0:
                continue

            min_x, min_y, width, height = bboxes[i_batch, j_label].tolist()
            max_x = min_x + width
            max_y = min_y + height

            bboxes_map[i_batch, j_label, min_y:max_y, min_x:max_x] = 1

    return bboxes_map


def create_sprite_atlas(dataset,
                        target_h=50,
                        target_w=50,
                        n_channels=3,
                        max_size=8192, # TB-frontend max-size available
                        ):
    """Creates an atlas of a dataset of images.

    Args:
        dataset -- Classification dataset or image iterator
    """
    n_images = len(dataset)

    def _resize_image(image):
        image = interpolate(image.unsqueeze(0), (target_h, target_w), mode='nearest')
        return image.squeeze(0)

    n_cols = int(np.ceil(np.sqrt(n_images)))
    n_rows = int(np.ceil(n_images / n_cols))

    total_height = target_h * n_rows
    total_width = target_w * n_cols
    if total_height > max_size or total_width > max_size:
        raise Exception(f'Atlas would be too big: {total_height}x{total_width}')

    atlas = torch.zeros(n_channels, total_height, total_width)

    for index, item in enumerate(dataset):
        if isinstance(item, BatchItem):
            image = item.image
        elif isinstance(item, torch.Tensor):
            image = item
        else:
            raise Exception(f'Item type not recognized: {type(item)}')

        image = tensor_to_range01(image) # shape: n_channels, height, width
        resized_image = _resize_image(image) # shape: n_channels, target_h, target_w

        row_from = (index // n_cols) * target_h
        row_to = row_from + target_h
        col_from = (index % n_cols) * target_w
        col_to = col_from + target_w

        atlas[:, row_from:row_to, col_from:col_to] = resized_image

    return atlas


def squeeze_masks(masks):
    """Squeezes organ masks to an image.

    Opposite of to_onehot

    Args:
        masks -- tensor of shape ([n_organs], height, width)
    Returns:
        masks as tensor of shape (height, width).
    """
    if isinstance(masks, numbers.Number) and masks == -1:
        return None

    if masks.ndim == 2:
        return masks

    n_organs = masks.size(0)
    multiplier = torch.arange(0, n_organs).unsqueeze(-1).unsqueeze(-1)
    return (multiplier * masks).sum(dim=0)


def load_image(image_fpath, image_format):
    try:
        image_fp = Image.open(image_fpath)
        image = image_fp.convert(image_format)
        image_fp.close()
    except OSError as e:
        LOGGER.error(
            'Failed to load image, may be broken: %s', image_fpath,
        )
        LOGGER.error(e)

        # FIXME: a way to ignore the image during training? (though it may break other things)
        raise

    return image

class MaskToTensor:
    def __init__(self, multilabel=False, n_labels=None):
        self.multilabel = multilabel
        self.n_labels = n_labels

        if multilabel:
            assert self.n_labels is not None, 'Cannot set multilabel=True and n_labels=None'

    def __call__(self, mask_pil):
        # NOTE: do not use transforms.ToTensor(),
        # as the range gets moved from 0..255 to 0..1
        mask = torch.from_numpy(np.array(mask_pil))
        # shape: height, width; type: uint8

        if self.multilabel:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
                # shape: 1, height, width

            mask = to_onehot(mask.long(), self.n_labels)
            # shape: 1, n_labels, height, width; type: uint8

            mask = mask.squeeze(0)
            # shape: n_labels, height, width


        # shape(seg_multilabel=True): n_labels, height, width
        # shape(seg_multilabel=False): height, width
        return mask

def get_default_mask_transform(image_size, seg_multilabel, n_seg_labels=None, crop_center=None):
    tfs = [
        transforms.Resize(image_size, Image.NEAREST),
    ]
    if crop_center is not None:
        tfs.append(transforms.CenterCrop(crop_center))
    tfs.append(MaskToTensor(seg_multilabel, n_seg_labels))
    return transforms.Compose(tfs)
