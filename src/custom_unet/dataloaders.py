from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import os
import numpy as np
import nibabel as nib


class CadpeDataset(Dataset):
    def __init__(self, img_dir, seg_dir, transform=None, with_frangi=False):
        self.transform = transform

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

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):
        image = nib.load(self.imgs_paths[idx]).get_data().astype(np.float32)
        seg = nib.load(self.segs_paths[idx]).get_data().astype(np.float32)
        if self.transform:
            image = self.transform(image)
            image = image[np.newaxis]
        if self.frangi_paths is not None:
            frangi = nib.load(self.frangi_paths[idx]).get_data().astype(np.float32)
            image = np.array([image, frangi])
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
