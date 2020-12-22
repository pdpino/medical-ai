import torch
import torch.nn as nn
from torch.nn.functional import interpolate

def downsample_target(target, size):
    return interpolate(target.float(), size, mode='nearest').long()

class OutOfTargetSumLoss(nn.Module):
    def forward(self, output, target, stops=None):
        """Computes the sum over the out-of-target attention scores.

        out-of-target: values where target is 0

        Args:
            output -- tensor of shape batch_size, n_sentences, height, width
            target -- tensor of shape batch_size, n_sentences, full-height, full-width
            stops -- tensor of shape (batch_size, n_sentences), indicating where each report stops.
                0 indicates the sentence is valid, 1 indicates the sentence is not valid (i.e. report stopped before)
        """
        height, width = output.size()[-2:]

        target = downsample_target(target, size=(height, width))
        # shape: batch_size, n_sentences, height, width

        out_of_target = (target == 0).long()
        # shape: batch_size, n_sentences, height, width

        wrong_values = -torch.log(1 - output * out_of_target) # Cross-entropy like
        # wrong_values = output * out_of_target # Raw sum
        # shape: batch_size, n_sentences, height, width

        # Sum out-of-target values
        wrong_values = wrong_values.sum(dim=(-2,-1))
        # shape: batch_size, n_sentences

        if stops is not None:
            ignore_filter = 1 - stops
            wrong_values = wrong_values * ignore_filter
            # shape: batch_size, n_sentences

        return torch.mean(wrong_values) # shape: 1