"""Create a UNet model to process 3D images for segmentation."""


from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DownLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: Tuple[int, int, int] = (2, 2, 2),
        pool_stride: Tuple[int, int, int] = (2, 2, 2),
        pool_type: Literal["max", "avg", "skip"] = "max",
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        if pool_type == "max":
            self.pool = nn.MaxPool3d(pool_size, stride=pool_stride)
        elif pool_type == "avg":
            self.pool = nn.AvgPool3d(pool_size, stride=pool_stride)
        elif pool_type == "skip":
            self.pool = nn.Identity()
        else:
            raise ValueError("Pool type must be either 'max' or 'avg'.")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.pool(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class UpLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up_sampling = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

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
    
    def forward(self, x: torch.Tensor, x_sym: torch.Tensor) -> torch.Tensor:
        x = self.up_sampling(x)
        x = self.crop_and_merge(x_sym, x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class CustomUNet(nn.Module):
    def __init__(
        self, 
        in_channels: int = 1, 
        internal_channels: int = 64, 
        n_layers: int = 4, 
        # patch_size: Tuple[int, int, int] = (50, 50, 50)
    ) -> None:
        super(CustomUNet, self).__init__()

        # self.patch_size = patch_size

        # Downsampling path
        self.downs = nn.ModuleList()
        for i in range(n_layers + 1):
            if i == 0:
                layer = DownLayer(in_channels, internal_channels, pool_type="skip")
            else:
                layer = DownLayer(internal_channels * 2**(i-1), internal_channels * 2**i)
            self.downs.append(layer)

        # Upsampling path
        self.ups = nn.ModuleList()
        for i in range(n_layers):
            layer = UpLayer(internal_channels * 2**(n_layers-i), internal_channels * 2**(n_layers-i-1))
            self.ups.append(layer)

        self.out_conv = nn.Conv3d(internal_channels, 1, kernel_size=1)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     x_shape = x.size()
    #     print(x_shape, self.patch_size)
    #     n_patches = [x_shape[i + 2] // self.patch_size[i] + 1 for i in range(3)]
    #     padded_x = torch.zeros(
    #         [x_shape[0], x_shape[1]] + [n_patches[i] * self.patch_size[i] for i in range(3)]
    #     ).to(x.device)
    #     x_slice = np.s_[:, :, 
    #         self.patch_size[0]//2: self.patch_size[0]//2 + x_shape[2],
    #         self.patch_size[1]//2: self.patch_size[1]//2 + x_shape[3],
    #         self.patch_size[2]//2: self.patch_size[2]//2 + x_shape[4],
    #     ]
    #     padded_x[x_slice] = x

    #     pred = torch.zeros_like(padded_x).to(x.device)
    #     for a in range(n_patches[0]):
    #         for b in range(n_patches[1]):
    #             for c in range(n_patches[2]):
    #                 patch_slice = np.s_[:,
    #                     a * self.patch_size[0] : (a + 1) * self.patch_size[0],
    #                     b * self.patch_size[1] : (b + 1) * self.patch_size[1],
    #                     c * self.patch_size[2] : (c + 1) * self.patch_size[2],
    #                 ]
    #                 patch = padded_x[patch_slice]
    #                 pred[patch_slice] = self.patch_forward(patch)
    #     return pred[x_slice]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Downsampling
        down_x = [x]
        for down_layer in self.downs:
            # print(down_x[-1].size())
            down_x.append(down_layer(down_x[-1]))

        # print(down_x[-1].size())
        # Upsampling
        up_x = self.ups[0](down_x[-1], down_x[-2])
        for i in range(1, len(self.ups)):
            # print(up_x.size())
            up_x = self.ups[i](up_x, down_x[-i-2])
        
        # print(up_x.size())
        out = self.out_conv(up_x)
        # print(out.size())
        return torch.sigmoid(out)
