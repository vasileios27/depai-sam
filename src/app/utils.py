import errno
import os
import logging
import shutil
import numpy as np
import rasterio


def create_mask(img, anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    mask = np.zeros((img.shape[:-1]), dtype=np.float32)
    for idx, ann in enumerate(sorted_anns):
        msk = ann["segmentation"].astype(np.uint8)
        mask[msk == 1] = idx + 1
    return mask


def clean_temp_directory(folder_name: str):
    try:
        shutil.rmtree(folder_name)
    except OSError as e:
        logging.error("Error: %s - %s.", e.filename, e.strerror)


def create_directory(folder_name: str):
    if not os.path.exists(folder_name):
        try:
            os.mkdir(folder_name)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def save_numpy_as_geotiff(img_array, template_file, output_file, count=1):
    if len(img_array.shape) == 2:
        count = 1
    elif len(img_array.shape) == 3:
        count = img_array.shape[-1]
    with rasterio.open(template_file) as src:
        template_profile = src.profile
    # Update the template profile with the array data
    template_profile.update(dtype=np.float32, count=count)
    # Create the output GeoTIFF file
    with rasterio.open(output_file, "w", **template_profile) as dst:
        dst.write(img_array, 1)
