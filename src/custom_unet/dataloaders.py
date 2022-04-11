import os

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


class CadpeDataset(Dataset):
    def __init__(
        self, img_dir, seg_dir, transform=None, with_frangi=False, patch_size=(50, 50, 50)
    ):
        self.transform = transform
        self.patch_size = patch_size
        self.patch_positions = self.compute_patch_positions()

        self.imgs_paths = [
            os.path.join(img_dir, img_name)
            for img_name in os.listdir(img_dir)
            if img_name.endswith(".nii.gz")
        ]
        self.imgs_paths.sort()
        self.segs_paths = [
            os.path.join(seg_dir, seg_name)
            for seg_name in os.listdir(seg_dir)
            if seg_name.endswith(".nii.gz")
        ]
        self.segs_paths.sort()

        self.frangi_paths = None
        if with_frangi:
            self.frangi_paths = [
                os.path.join(seg_dir, seg_name)
                for seg_name in os.listdir(seg_dir)
                if seg_name.endswith(".nii.gz")
            ].sort()

    def compute_patch_positions(self):
        patch_positions = []
        for i, img_path in enumerate(self.imgs_paths):
            img = nib.load(img_path)
            img_shape = img.get_data_shape()
            patch_positions.extend(
                [
                    (i, (a, b, c))
                    for a in range(img_shape[0] // self.patch_size[0] - 1)
                    for b in range(img_shape[1] // self.patch_size[1] - 1)
                    for c in range(img_shape[2] // self.patch_size[2] - 1)
                ]
            )

        return patch_positions

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, patch_idx):
        img_idx, patch_pos = self.patch_positions[patch_idx]
        patch_slice = np.s_[
            patch_pos[0] : patch_pos[0] + self.patch_size[0],
            patch_pos[1] : patch_pos[1] + self.patch_size[1],
            patch_pos[2] : patch_pos[2] + self.patch_size[2],
        ]
        image = nib.load(self.imgs_paths[img_idx]).get_data().astype(np.float32)[patch_slice]
        seg = nib.load(self.segs_paths[img_idx]).get_data().astype(np.float32)[patch_slice]
        if self.transform:
            image = self.transform(image)
        if self.frangi_paths is not None:
            frangi = nib.load(self.frangi_paths[img_idx]).get_data().astype(np.float32)[patch_slice]
            image = np.array([image, frangi])
        else:
            image = image[np.newaxis]
        return image, seg


def get_data_loaders(
    train_img_path,
    train_seg_path,
    test_img_path="",
    test_seg_path="",
    batch_size=1,
    val_proportion: float = 0.1,
    with_frangi=False,
):
    """Get the data loaders"""
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
    )

    train_dataset = CadpeDataset(
        train_img_path, train_seg_path, transform=train_transform, with_frangi=with_frangi
    )

    val_dataset_size = int(len(train_dataset) * val_proportion)
    train_dataset, val_dataset = random_split(
        train_dataset,
        [len(train_dataset) - val_dataset_size, val_dataset_size],
        generator=torch.Generator().manual_seed(42),
    )

    if test_img_path:
        test_dataset = CadpeDataset(
            test_img_path, test_seg_path, transform=test_transform, with_frangi=with_frangi
        )
        test_loader = DataLoader(dataset=test_dataset, shuffle=False)
    else:
        test_loader = None

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, shuffle=False)

    return train_loader, val_loader, test_loader
