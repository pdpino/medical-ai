import torch
import torch.nn as nn

class _ResBlock(nn.Module):
    def __init__(self, n_channels, kernel):
        super().__init__()

        stride = (1, 1)
        padding = (kernel - 1) // 2 # assure output resolution = input resolution

        assert kernel % 2 == 1, f'kernel should be odd, got: {kernel}'

        self.block = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel, stride, padding=padding),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, kernel, stride, padding=padding),
            nn.BatchNorm2d(n_channels),
        )

        self.final = nn.Sequential(
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        # all shapes: batch_size, n_channels, height, width
        y = self.block(x)
        z = self.final(x + y)
        return z


class _ParallelResBlocks(nn.Module):
    def __init__(self, n_parallel, n_channels, kernel):
        super().__init__()

        self.res_blocks = nn.ModuleList([
            _ResBlock(n_channels, kernel)
            for _ in range(n_parallel)
        ])

    def forward(self, x):
        # x shape: batch_size, n_channels, height, width
        y = [
            res_block(x) # shape: batch_size, n_channels, height, width
            for res_block in self.res_blocks
        ]

        y = torch.cat(y, dim=1)
        # shape: batch_size, n_channels * n_parallel, height, width

        return y

class ScanFCN(nn.Module):
    """(Attempting to) reproduce model from:
    SCAN: Structure Correcting Adversarial Network for Organ Segmentation in Chest X-Rays.
    """
    def __init__(self, n_classes=4, **unused_kwargs):
        super().__init__()

        self.fcn = nn.Sequential(
            _ParallelResBlocks(8, 1, 7), # output: 8 x 400 x 400
            _ResBlock(8, 3),
            nn.AvgPool2d((2, 2), (2, 2)), # output: 8 x 200 x 200
            _ParallelResBlocks(2, 8, 3), # output: 16 x 200 x 200
            nn.AvgPool2d((2, 2), (2, 2)), # output: 16 x 100 x 100
            _ParallelResBlocks(2, 16, 3), # output: 32 X 100 x 100
            nn.AvgPool2d((2, 2), (2, 2)), # output: 32 x 50 x 50
            _ParallelResBlocks(2, 32, 3), # output: 64 x 50 x 50
            nn.AvgPool2d((2, 2), (2, 2)), # output: 64 x 25 x 25
            _ResBlock(64, 1),
            _ResBlock(64, 3),
            _ResBlock(64, 1),
            _ResBlock(64, 3),
            _ResBlock(64, 1), # output: 64 x 25 x 25
            nn.Conv2d(64, 4, 1, 1), # output: 4 x 25 x 25
            nn.ConvTranspose2d(4, n_classes, 32, 16, padding=8), # output: 4 x 400 x 400
        )

    def forward(self, x):
        # x shape: batch_size, 1, height, width
        return self.fcn(x) # shape: batch_size, n_classes, height, width
