import torch
from torchvision import transforms
from tqdm.auto import tqdm


def compute_mean_std(image_iterator, show=False):
    """Computes mean and std of a dataset.

    Args:
      image_iterator -- should yield one image at the time; each image is a tensor of shape (n_channels=3, height, width)

    Returns:
      Channel wise mean, std (i.e. tensors of shape n_channels)
    """
    n_channels = 3
    mean = torch.zeros(n_channels)
    std = torch.zeros(n_channels)

    n_samples = 0

    if show:
        image_iterator = tqdm(image_iterator)

    for image in image_iterator:
        image_flatten = image.view(image.size(0), -1)

        mean += image_flatten.mean(1)
        std += image_flatten.std(1)

        n_samples += 1

    mean /= n_samples
    std /= n_samples

    return mean, std



class NormalizeBySample:
    """Normalizes images with their mean and std."""
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon

    def __call__(self, image):
        n_channels, height, width = image.size()
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