import torch
import torch.nn as nn
import torch.nn.functional as F


def round_(filter):
    return round(filter / 3)


class ResidualBranch3D(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size, stride, branch_index):
        super(ResidualBranch3D, self).__init__()

        # Adjust filter_size, stride, and padding for 3D
        filter_size = (
            filter_size,
            filter_size,
            filter_size,
        )  # Example adjustment for 3D
        stride = (stride, stride, stride)
        padding = round_(filter_size[0])  # Simple round function for padding

        self.residual_branch = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=filter_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=filter_size,
                stride=1,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x):
        return self.residual_branch(x)


class SkipConnection3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(SkipConnection3D, self).__init__()

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.skip_layers = nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            ),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x):
        x = F.interpolate(
            x, scale_factor=self.stride, mode="trilinear", align_corners=False
        )
        x = self.skip_layers(x)
        return x


class ResNet3DSegment(nn.Module):
    def __init__(self, num_classes):
        super(ResNet3DSegment, self).__init__()

        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        # Residual blocks
        self.resblock1 = ResidualBranch3D(64, 128, 3, 1, 1)
        self.resblock2 = ResidualBranch3D(128, 256, 3, 2, 2)
        self.resblock3 = ResidualBranch3D(256, 512, 3, 2, 3)

        # Upsampling layers
        self.upsample1 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)

        # Output layer
        self.out_conv = nn.Conv3d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.initial(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.out_conv(x)
        return x


# Example usage:
model = ResNet3DSegment(num_classes=3)
input_tensor = torch.rand(1, 1, 64, 64, 64)  # Example 3D input
output = model(input_tensor)
print(output.shape)
