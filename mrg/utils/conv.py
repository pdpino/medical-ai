import torch
from torch import nn

def _calc_output_size(input_size, kernel_size, stride, dilation=0, padding=0):
    """Calculates output size for a given convolution configuration.
    
    Should work with Conv and MaxPool layers.
    See formula in docs https://pytorch.org/docs/stable/nn.html#conv2d
    """
    if type(input_size) is not torch.Tensor:
        input_size = torch.tensor(input_size)

    kernel_size = torch.tensor(kernel_size)
    stride = torch.tensor(stride)
    dilation = torch.tensor(dilation)
    padding = torch.tensor(padding)

    value = (input_size + 2*padding - dilation * (kernel_size - 1) - 1)
    value = value.true_divide(stride)
    value += 1

    return torch.floor(value.float()).long()


def calc_module_output_size(model, input_size):
    """Calculates output size of a model.
    
    Considers only Conv2d and MaxPool2d layers.
    Tested only with Sequential layers, deeper configurations may not work
    """
    last_channel_out = None

    size = input_size
    for submodule in model.modules():
        if type(submodule) is nn.Conv2d or type(submodule) is nn.MaxPool2d:
            size = _calc_output_size(size, submodule.kernel_size, submodule.stride,
                                     dilation=submodule.dilation,
                                     padding=submodule.padding,
                                     )

        if type(submodule) is nn.Conv2d:
            last_channel_out = submodule.out_channels

    return last_channel_out, size