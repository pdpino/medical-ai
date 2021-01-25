import os
import torch
from torchvision import transforms
from tqdm.auto import tqdm
from PIL import Image


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


def compute_mean_std(image_iterator, n_channels=3, show=False):
    """Computes mean and std of a dataset.

    Args:
      image_iterator -- should yield one image at the time; each image is a tensor of shape (n_channels=3, height, width)

    Returns:
      Channel wise mean, std (i.e. tensors of shape n_channels)
    """
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


def get_default_image_transform(image_size=(512, 512),
                                norm_by_sample=True,
                                mean=0,
                                std=1,
                                ):
    def _to_channel_list(value):
        if isinstance(value, (list, tuple)):
            return value
        return [value]

    if norm_by_sample:
        norm_transform = NormalizeBySample()
    else:
        mean = _to_channel_list(mean)
        std = _to_channel_list(std)
        norm_transform = transforms.Normalize(mean, std)

    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        norm_transform,
    ])


def bbox_coordinates_to_map(bboxes, valid, image_size):
    """Transform bbox (x,y,w,h) pairs to binary maps.

    Args:
      bboxes - tensor of shape (batch_size, n_diseases, 4)
      valid - tensor of shape (batch_size, n_diseases)
    Returns:
      tensor of shape (batch_size, n_diseases, height, width)
    """
    batch_size, n_labels = valid.size()
    bboxes_map = torch.zeros(batch_size, n_labels, *image_size).to(bboxes.device)

    for i_batch in range(batch_size):
        for j_label in range(n_labels):
            if valid[i_batch, j_label].item() == 0:
                continue

            min_x, min_y, width, height = bboxes[i_batch, j_label].tolist()
            max_x = min_x + width
            max_y = min_y + height

            bboxes_map[i_batch, j_label, min_y:max_y, min_x:max_x] = 1

    return bboxes_map
