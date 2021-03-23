import torch


def threshold_attributions(attributions, thresh=0.5):
    """Apply a threshold over attributions.

    Args:
        attributions -- tensor of any shape, with values between 0 and 1
        threshold -- float to apply the threshold
    Returns:
        thresholded attributions, same shape as input
    """
    device = attributions.device
    size = attributions.size()

    ones = torch.ones(size, device=device)
    zeros = torch.zeros(size, device=device)
    attributions = torch.where(attributions >= thresh, ones, zeros)

    return attributions