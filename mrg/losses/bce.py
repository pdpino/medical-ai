import torch
from torch import nn
import numpy as np

class WeigthedBCELoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        
        self.epsilon = epsilon

    def forward(self, output, target):
        """Computes weighted binary cross entropy loss.
        
        If a multi-label array is given, the BCE is summed across labels.
        Note that the BP and BN weights are calculated by batch, not in the whole dataset.
        """
        output = output.clamp(min=self.epsilon, max=1-self.epsilon)
        target = target.float()

        # Calculate weights
        BP = 1
        BN = 1

        total = np.prod(target.size())
        positive = (target == 1).sum().item()
        negative = total - positive

        if positive != 0 and negative != 0:
            BP = total / positive
            BN = total / negative

        loss = -BP * target * torch.log(output) - BN * (1 - target) * torch.log(1 - output)

        return torch.sum(loss)


class WeigthedBCEByDiseaseLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super().__init__()

        self.epsilon = epsilon

    def forward(self, output, target):
        output = output.clamp(min=self.epsilon, max=1-self.epsilon)
        target = target.float()

        batch_size, n_diseases = target.size()
        
        # total = torch.Tensor().new_full((n_diseases,), batch_size).type(torch.float)
        positive = torch.sum(target == 1, dim=0).type(torch.float)
        negative = torch.sum(target == 0, dim=0).type(torch.float)
        total = positive + negative
        # shapes: n_diseases


        # If a value is zero, is set to batch_size (so the division results in 1 for that disease)
        positive = positive + ((positive == 0)*batch_size).type(positive.dtype)
        negative = negative + ((negative == 0)*batch_size).type(negative.dtype)
        
        BP = total / positive
        BN = total / negative
        
        loss = -BP * target * torch.log(output) - BN * (1 - target) * torch.log(1 - output)

        return torch.sum(loss)
