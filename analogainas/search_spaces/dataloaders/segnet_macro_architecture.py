import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBranch(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size, stride, branch_index):
        super(ResidualBranch, self).__init__()
        self.residual_branch = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=filter_size,
                stride=stride,
                padding=filter_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=filter_size,
                stride=1,
                padding=filter_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.residual_branch(x)


class ResidualGroup(nn.Module):
    def __init__(
        self,
        block,
        in_channels,
        out_channels,
        num_blocks,
        filter_size,
        num_branches,
        stride=1,
    ):
        super(ResidualGroup, self).__init__()
        self.group = nn.Sequential()
        for i in range(num_blocks):
            self.group.add_module(
                "Block_{}".format(i + 1),
                block(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    filter_size,
                    stride if i == 0 else 1,
                    num_branches,
                ),
            )

    def forward(self, x):
        return self.group(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.upsample(x)


class SegmentNet(nn.Module):
    def __init__(self, input_dim, classes, widen_factors=[1, 1.5, 2, 2.5]):
        super(SegmentNet, self).__init__()
        self.widen_factors = widen_factors
        initial_channels = 64
        self.blocks = nn.Sequential()

        current_channels = initial_channels
        for i, widen_factor in enumerate(widen_factors):
            next_channels = int(initial_channels * widen_factor)
            self.blocks.add_module(
                f"ResidualGroup{i}",
                ResidualGroup(
                    ResidualBranch,
                    current_channels,
                    next_channels,
                    3,
                    3,
                    1,
                    stride=2 if i == 0 else 1,
                ),
            )
            current_channels = next_channels

        self.upsample1 = UpsampleBlock(current_channels, current_channels // 2)
        self.upsample2 = UpsampleBlock(current_channels // 2, current_channels // 4)
        self.upsample3 = UpsampleBlock(current_channels // 4, current_channels // 8)

        self.final_conv = nn.Conv2d(current_channels // 8, classes, kernel_size=1)

    def forward(self, x):
        x = self.blocks(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.final_conv(x)
        return x
