import logging
import torch.nn as nn
from torch.nn.functional import interpolate, binary_cross_entropy

LOGGER = logging.getLogger(__name__)

def _interpolate(target, size):
    return interpolate(target, size, mode='nearest')

class OutOfTargetSumLoss(nn.Module):
    def __init__(self, interpolate_target=True, eps=1e-8):
        """Constructor.

        Args:
            interpolate_target -- If true, resize the GT when GT and generated sizes
                do not match. If false, resize the generated output.
        """
        super().__init__()

        self.eps = eps

        self._interpolate_target = interpolate_target

    def forward(self, output, target, stops=None):
        """Computes the sum over the out-of-target attention scores.

        out-of-target: values where target is 0

        Args:
            output -- FloatTensor of shape (batch_size, n_channels, output_h, output_w)
            target -- LongTensor of shape (batch_size, n_channels, target_h, target_h)
            stops -- tensor of shape (batch_size, n_channels), indicating where each report stops.
                0 indicates the sentence is valid, 1 indicates the sentence
                is not valid (i.e. report stopped before).

        Returns:
            Calculated loss

        Notice n_channels dimension may be anything, such as n_diseases, n_sentences, etc
        """
        output_size = output.size()[-2:]
        target_size = target.size()[-2:]

        if output_size != target_size:
            if self._interpolate_target:
                target = _interpolate(target.float(), size=output_size).long()
                # shape: batch_size, n_channels, output_h, output_w
            else:
                output = _interpolate(output, size=target_size)
                # shape: batch_size, n_channels, target_h, target_w

        loss = binary_cross_entropy(output, target.float(), reduction='none')
        # shape: bs, n_channels, height, width

        filter_values = (target == 0)
        if stops is not None:
            stops = stops.unsqueeze(-1).unsqueeze(-1) # shape: bs, n_channels, 1, 1
            filter_values &= (stops == 0)

        loss = loss[filter_values]
        if loss.size() == (0,):
            return loss.sum() # return a 0

        return loss.mean()

        # DELETEME: old (wrong) version
        # out_of_target = (target == 0).long()
        # # shape: batch_size, n_channels, height, width

        # wrong_values = -torch.log(1 - output * out_of_target + self.eps) # Cross-entropy like
        # # wrong_values = output * out_of_target # Raw sum
        # # shape: batch_size, n_channels, height, width

        # # Sum out-of-target values
        # wrong_values = wrong_values.sum(dim=(-2,-1))

        # # Mean over pixels out-of-target pixels
        # n_pixels_out_of_target = out_of_target.sum(dim=(-2,-1)) # (bs, n_channels)
        # wrong_values = divide_tensors(wrong_values, n_pixels_out_of_target) # (bs, n_channels)

        # if stops is not None:
        #     ignore_filter = 1 - stops
        #     wrong_values = wrong_values * ignore_filter
        #     # shape: batch_size, n_channels

        # return torch.mean(wrong_values) # shape: 1
