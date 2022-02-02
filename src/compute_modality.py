import os
import nibabel as nib
from src.frangi import compute_frangi
from scancovia import AiSegment


def compute_and_save_modality(input_filename, output_filename, force=False):
    if not force and os.exists(output_filename):
        return

    image = nib.load(input_filename)
    image_data = image.get_fdata()

    frangi_mask = compute_frangi(image_data)

    ai_segment = AiSegment(device="cuda")
    output = ai_segment(input_filename)
    lung_mask = output["lungs_mask"].transpose(1, 0, 2)

    modality = frangi_mask * lung_mask

    new_image = nib.Nifti1Image(modality, image.affine, image.header)
    nib.save(new_image, output_filename)
