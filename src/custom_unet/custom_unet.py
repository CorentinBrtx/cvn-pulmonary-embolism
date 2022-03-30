"""Create a UNet model to process 3D images for segmentation."""


from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: Tuple[int, int, int] = (2, 2, 2),
        pool_stride: Tuple[int, int, int] = (2, 2, 2),
        pool_type: Literal["max", "avg"] = "max",
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3)
        self.bn2 = nn.BatchNorm3d(out_channels)

        if pool_type == "max":
            self.pool = nn.MaxPool3d(pool_size, stride=pool_stride)
        elif pool_type == "avg":
            self.pool = nn.AvgPool3d(pool_size, stride=pool_stride)
        else:
            raise ValueError("Pool type must be either 'max' or 'avg'.")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x_pooled = self.pool(x)
        return x_pooled, x


class UpLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.up_sampling = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.up_sampling(x)
        return x


class CustomUNet(nn.Module):
    def __init__(self, in_channels: int = 1, internal_channels: int = 64):
        super(CustomUNet, self).__init__()

        # Downsampling path
        self.down1 = DownLayer(in_channels, internal_channels)
        self.down2 = DownLayer(internal_channels, internal_channels * 2)
        self.down3 = DownLayer(internal_channels * 2, internal_channels * 4)
        self.down4 = DownLayer(internal_channels * 4, internal_channels * 8)

        # Upsampling path
        self.up1 = UpLayer(internal_channels * 8, internal_channels * 16)
        self.up2 = UpLayer(internal_channels * 16, internal_channels * 8)
        self.up3 = UpLayer(internal_channels * 8, internal_channels * 4)
        self.up4 = UpLayer(internal_channels * 4, internal_channels * 2)

        self.conv1 = nn.Conv3d(internal_channels * 2, internal_channels, kernel_size=3)
        self.conv2 = nn.Conv3d(internal_channels, internal_channels, kernel_size=3)
        self.conv3 = nn.Conv3d(internal_channels, 1, kernel_size=1)

    @staticmethod
    def crop_and_merge(
        from_down: torch.Tensor, from_up: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ds = from_down.shape[2:]
        us = from_up.shape[2:]
        from_down = from_down[
            :,
            :,
            ((ds[0] - us[0]) // 2) : ((ds[0] + us[0]) // 2),
            ((ds[1] - us[1]) // 2) : ((ds[1] + us[1]) // 2),
            ((ds[2] - us[2]) // 2) : ((ds[2] + us[2]) // 2),
        ]
        merged = torch.cat((from_down, from_up), 1)
        return merged

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Downsampling
        down1_pooled, down1 = self.down1(x)
        down2_pooled, down2 = self.down2(down1_pooled)
        down3_pooled, down3 = self.down3(down2_pooled)
        # down4_pooled, down4 = self.down4(down3_pooled)

        # Upsampling
        # up1 = self.up1(down4_pooled)
        # up2 = self.up2(self.crop_and_merge(down4, up1))

        up2 = self.up2(down3_pooled)
        up3 = self.up3(self.crop_and_merge(down3, up2))
        up4 = self.up4(self.crop_and_merge(down2, up3))

        out1 = self.conv1(self.crop_and_merge(down1, up4))
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        return out3
