import os
import numpy as np

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, random_split
from torchvision import transforms
from typing import Tuple, List


class CadpeDatasetBase:
    def __init__(
        self,
        imgs_paths: List[str],
        segs_paths: List[str],
        transform=None,
        with_frangi: List[str] = None,
        patch_size: Tuple[int, int, int] = (50, 50, 50)
    ):
        self.transform = transform
        self.patch_size = patch_size

        self.imgs_paths = imgs_paths

        self.segs_paths = segs_paths

        self.frangi_paths = None
        if with_frangi:
            self.frangi_paths = with_frangi

    def get_patch_slice(self, patch_idx: Tuple[int, int, int], idx_starts: Tuple[int, int, int]):
        a, b, c = patch_idx
        return np.index_exp[
            idx_starts[0] + a * (self.patch_size[0]) : idx_starts[0] + (a + 1) * self.patch_size[0],
            idx_starts[1] + b * (self.patch_size[1]) : idx_starts[1] + (b + 1) * self.patch_size[1],
            idx_starts[2] + c * (self.patch_size[2]) : idx_starts[2] + (c + 1) * self.patch_size[2],
        ]


class CadpeDataset(CadpeDatasetBase, Dataset):
    def __init__(
        self,
        imgs_paths: List[str],
        segs_paths: List[str],
        transform=None,
        with_frangi: List[str] = None,
        patch_size: Tuple[int, int, int] = (50, 50, 50)
    ):
        super().__init__(imgs_paths, segs_paths, transform, with_frangi, patch_size)
        self.patch_positions = self.compute_patch_positions()

    def compute_patch_positions(self):
        patch_positions = []
        for i, img_path in enumerate(self.imgs_paths):
            img = nib.load(img_path)
            img_shape = img.header.get_data_shape()
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
        return len(self.patch_positions)

    def __getitem__(self, patch_idx: int):
        img_idx, (a, b, c) = self.patch_positions[patch_idx]
        patch_slice = self.get_patch_slice((a, b, c))
        image = nib.load(self.imgs_paths[img_idx]).get_data().astype(np.float32)[patch_slice]
        seg = nib.load(self.segs_paths[img_idx]).get_data().astype(np.float32)[patch_slice]
        if self.frangi_paths is not None:
            frangi = nib.load(self.frangi_paths[img_idx]).get_data().astype(np.float32)[patch_slice]
            image = np.array([image, frangi])
        else:
            image = image[np.newaxis]

        if self.transform:
            image = self.transform(image)
            seg = self.transform(seg)
        return torch.from_numpy(image), torch.from_numpy(seg)


class IterableCadpeDataset(CadpeDatasetBase, IterableDataset):
    def __init__(
        self,
        imgs_paths: List[str],
        segs_paths: List[str],
        transform=None,
        with_frangi: List[str] = None,
        patch_size: Tuple[int, int, int] = (50, 50, 50),
        seed: int = 42
    ):
        super().__init__(imgs_paths, segs_paths, transform, with_frangi, patch_size)
        self.random = np.random.default_rng(seed)

    def __len__(self):
        """This is an estimation"""
        return len(self.imgs_paths) * (512 // self.patch_size[0])**3

    def __iter__(self):
        # shuffle_idx = self.random.permutation(len(self.imgs_paths))
        uid = torch.utils.data.get_worker_info().id
        worker_total_num = torch.utils.data.get_worker_info().num_workers
        start = uid * len(self.imgs_paths) // worker_total_num
        end = min(len(self.imgs_paths), (uid + 1) * len(self.imgs_paths) // worker_total_num)
        for idx in range(start, end):
            img_data = nib.load(self.imgs_paths[idx]).get_data().astype(np.float32)
            if self.frangi_paths is not None:
                frangi_data = nib.load(self.frangi_paths[idx]).get_data().astype(np.float32)
                img_data = np.array([img_data, frangi_data])
            else:
                img_data = img_data[np.newaxis]

            shape = img_data.shape[1:]
            n_patches = [shape[i] // self.patch_size[i] for i in range(3)]
            slice_starts = [
                (shape[i] % self.patch_size[i]) // 2 for i in range(3)
            ]
            seg_data = nib.load(self.segs_paths[idx]).get_data().astype(np.float32)

            if self.transform:
                img_data = self.transform(img_data)
                seg_data = self.transform(seg_data)

            for a in range(n_patches[0]):
                for b in range(n_patches[1]):
                    for c in range(n_patches[2]):
                        patch_slice = self.get_patch_slice((a, b, c), slice_starts)
                        image = img_data[(slice(2),) + patch_slice]
                        seg = seg_data[patch_slice]
                        yield torch.from_numpy(image), torch.from_numpy(seg)


def search_imgs(img_dir: str):
    imgs_paths = [
                os.path.join(img_dir, img_name)
                for img_name in os.listdir(img_dir)
                if img_name.endswith(".nii.gz")
            ]
    imgs_paths.sort()
    return(imgs_paths)


def get_data_loaders(
    train_img_path: str,
    train_seg_path: str,
    test_img_path: str = "",
    test_seg_path: str = "",
    batch_size: int =1,
    patch_size: Tuple[int, int, int] = (50, 50, 50),
    val_proportion: float = 0.1,
    with_frangi: List[str] = None,
    iterable: bool = False,
    num_workers: int = 0,
    seed: int = 42,
):
    """Get the data loaders"""
    imgs_paths = search_imgs(train_img_path)
    segs_paths = search_imgs(train_seg_path)
    val_dataset_size = int(len(imgs_paths) * val_proportion)

    if iterable:
        random = np.random.default_rng(seed)
        shuffle = random.permutation(len(imgs_paths))
        train_dataset = IterableCadpeDataset(
            [imgs_paths[i] for i in shuffle[val_dataset_size:]],
            [segs_paths[i] for i in shuffle[val_dataset_size:]],
            # transform=train_transform,
            with_frangi=with_frangi,
            patch_size=patch_size,
        )
        val_dataset = IterableCadpeDataset(
            [imgs_paths[i] for i in shuffle[:val_dataset_size]],
            [segs_paths[i] for i in shuffle[:val_dataset_size]],
            # transform=train_transform,
            with_frangi=with_frangi,
            patch_size=patch_size,
        )
    else:
        train_dataset = CadpeDataset(
            imgs_paths,
            segs_paths,
            # transform=train_transform,
            with_frangi=with_frangi,
            patch_size=patch_size
        )

        train_dataset, val_dataset = random_split(
            train_dataset,
            [len(train_dataset) - val_dataset_size, val_dataset_size],
            generator=torch.Generator().manual_seed(seed),
        )

    # set num_workers to 0 when iterable so that the data loader doesn't fetch all dataset at once
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=not iterable,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        shuffle=False,
        num_workers=num_workers
    )

    if test_img_path:
        test_dataset = (IterableCadpeDataset if iterable else CadpeDataset)(
            search_imgs(test_img_path),
            search_imgs(test_seg_path),
            # transform=test_transform,
            with_frangi=with_frangi,
            patch_size=patch_size
        )
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, num_workers=num_workers)
    else:
        test_loader = None

    return train_loader, val_loader, test_loader
