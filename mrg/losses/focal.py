import torch
from torch import nn
from torch.nn.functional import softmax

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, epsilon=1e-5):
        """Computes focal loss.
        
        If a multi-label array is given, the loss is summed across labels.
        Based on this post: https://leimao.github.io/blog/Focal-Loss-Explained/
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, output, target):
        output = torch.sigmoid(output) # TODO: handle multilabel=False

        output = output.clamp(min=self.epsilon, max=1-self.epsilon)
        target = target.float()

        # Calculate p_t
        # Note that (for each label) only one term will survive, either output or (1-output)
        pt = target * output + (1 - target) * (1 - output)
        
        # Calculate log(p_t)
        # It could also be calculated as torch.log(pt)
        log_pt = target * torch.log(output) + (1 - target) * torch.log(1 - output)

        # Calculate other terms
        alpha_t = target * self.alpha + (1 - target)*(1 - self.alpha) # Only one term survives
        term_gamma = (1 - pt) ** self.gamma
        
        loss = - alpha_t * term_gamma * log_pt

        return torch.sum(loss)
