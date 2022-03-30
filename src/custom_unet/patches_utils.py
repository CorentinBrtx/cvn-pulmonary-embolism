"""Utils to cut input 3D images in smaller patches and merge output segmentation parches"""
import numpy as np


def cut_image(image, patch_size, stride):
    """Cut image in patches of size patch_size and stride stride"""
    # Get image size
    image_size = image.shape
    # Get number of patches
    nb_patches_x = int((image_size[0] - patch_size[0]) / stride) + 1
    nb_patches_y = int((image_size[1] - patch_size[1]) / stride) + 1
    nb_patches_z = int((image_size[2] - patch_size[2]) / stride) + 1
    # Get patches
    patches = np.zeros(
        (nb_patches_x, nb_patches_y, nb_patches_z, patch_size[0], patch_size[1], patch_size[2])
    )
    for i in range(nb_patches_x):
        for j in range(nb_patches_y):
            for k in range(nb_patches_z):
                patches[i, j, k, :, :, :] = image[
                    i * stride : i * stride + patch_size[0],
                    j * stride : j * stride + patch_size[1],
                    k * stride : k * stride + patch_size[2],
                ]
    return patches


def merge_segmentation(segmentation, patch_size, stride):
    """Merge segmentation patches into an image"""
    # Get image size
    image_size = segmentation.shape
    # Get number of patches
    nb_patches_x = int((image_size[0] - patch_size[0]) / stride) + 1
    nb_patches_y = int((image_size[1] - patch_size[1]) / stride) + 1
    nb_patches_z = int((image_size[2] - patch_size[2]) / stride) + 1
    # Get patches
    merged_segmentation = np.zeros(
        (nb_patches_x * patch_size[0], nb_patches_y * patch_size[1], nb_patches_z * patch_size[2])
    )
    for i in range(nb_patches_x):
        for j in range(nb_patches_y):
            for k in range(nb_patches_z):
                merged_segmentation[
                    i * stride : i * stride + patch_size[0],
                    j * stride : j * stride + patch_size[1],
                    k * stride : k * stride + patch_size[2],
                ] = segmentation[i, j, k]
    return merged_segmentation
