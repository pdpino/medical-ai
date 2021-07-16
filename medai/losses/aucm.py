import torch
from torch import nn


## Copied (and improved) from https://github.com/yzhuoning/LibAUC
class AUCMLoss(nn.Module):
    """
    AUCM Loss: a novel loss function to directly optimize AUROC

    inputs:
        margin: margin term for AUCM loss, e.g., m in [0, 1]
        imratio: imbalance ratio, i.e.,
            the ratio of number of postive samples to number of total samples
    outputs:
        loss value

    Reference:
        Yuan, Z., Yan, Y., Sonka, M. and Yang, T., 2020.
        Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical
            Image Classification.
        arXiv preprint arXiv:2012.03173.
    Link:
        https://arxiv.org/abs/2012.03173
    """
    def __init__(self, margin=1.0, imratio=None, device='cuda'):
        super(AUCMLoss, self).__init__()
        self.margin = margin
        self.p = imratio
        self.a = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True)
        self.b = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True)
        self.alpha = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True)

    def forward(self, y_pred, y_true):
        if self.p is None:
            self.p = (y_true==1).float().sum()/y_true.shape[0]

        y_pred = y_pred.reshape(-1, 1) # be carefull about these shapes
        y_true = y_true.reshape(-1, 1)
        loss = (1-self.p)*torch.mean((y_pred - self.a)**2*(y_true == 1).float()) + \
                    self.p*torch.mean((y_pred - self.b)**2*(y_true == 0).float())   + \
                    2*self.alpha*(self.p*(1-self.p)*self.margin + \
                    torch.mean((self.p*y_pred*(y_true == 0).float() - \
                    (1-self.p)*y_pred*(y_true == 1).float())) )- \
                    self.p*(1-self.p)*self.alpha**2
        return loss
