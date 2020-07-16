import torch
from torch import nn
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, epsilon=1e-5, multilabel=True, reduction='mean'):
        """Computes focal loss.
        
        If a multi-label array is given, the loss is averaged across labels.
        Based on:
        https://leimao.github.io/blog/Focal-Loss-Explained/
        https://discuss.pytorch.org/t/\
            focal-loss-for-imbalanced-multi-class-classification-in-pytorch/61289/2
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.multilabel = multilabel
        
        if multilabel:
            self._forward = self.forward_multilabel
        else:
            self._forward = self.forward_multiclass
            
        if reduction == 'mean':
            self._reduction = torch.mean
        elif reduction == 'sum':
            self._reduction = torch.sum
        else:
            self._reduction = lambda x: x
        
    def forward(self, *args, **kwargs):
        return self._forward(*args, **kwargs)
        
    def forward_multiclass(self, output, target):
        """Calculates focal loss for a multiclass (and single-label) classification problem.
        
        Args:
            output -- tensor of shape batch_size, n_labels (raw scores, without softmax)
            target -- tensor of shape batch_size (containing indices)
        """
        ce_loss = cross_entropy(output, target, reduction='none')
        # shape: batch_size

        # FIXME: more direct way of calculating pt?
        # i.e. applying softmax + one-hot encode target + calculate:
        # pt = target * output + (1 - target) * (1 - output) # shape: batch_size, n_labels
        # pt = pt.sum(dim=-1) # shape: batch_size
        pt = torch.exp(-ce_loss)
        # shape: batch_size
        
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        # shape: batch_size
        
        return self._reduction(focal_loss)
        
    def forward_multilabel(self, output, target):
        """Calculates focal loss for a multilabel classification problem.
        
        Args:
            output -- tensor of shape batch_size, n_labels
            target -- tensor of shape batch_size, n_labels
        """
        target = target.float()
        # Calculate log_pt and pt using functional cross entropy
        log_pt = binary_cross_entropy_with_logits(output, target, reduction='none')
        # shape: batch_size, n_labels

        pt = torch.exp(-log_pt)
        # shape: batch_size, n_labels
        
        # Calculate other terms
        alpha_t = target * self.alpha + (1 - target)*(1 - self.alpha) # Only one term survives
        # shape: batch_size, n_labels

        term_gamma = (1 - pt) ** self.gamma
        # shape: batch_size, n_labels
        
        loss = alpha_t * term_gamma * log_pt
        # shape: batch_size, n_labels

        return self._reduction(loss)