import os
import logging
from typing import Dict, List
import uuid
import torch
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import tifffile as tiff
from sam.config import TORCH_DEVICE, MODEL_TYPE, WEIGHTS_PATH
from sam.sam.build_sam import sam_model_registry
from sam.sam.predictor import SamPredictor
from app.utils import save_numpy_as_geotiff


def segment_sam_prompt(list_images: List[Dict]) -> List[Dict]:
    task_id = str(uuid.uuid4())
    logging.info("SAM Model Initialization")
    _ = torch.device(TORCH_DEVICE)
    sam = sam_model_registry[MODEL_TYPE](checkpoint=WEIGHTS_PATH)
    predictor = SamPredictor(sam)
    temp_path = os.path.join(os.path.dirname(list_images[0]["image_path"]), task_id)
    os.makedirs(temp_path, exist_ok=True)
    processing_status = []
    for image in list_images:
        res_output = {
            "image_uri": image["image_path"],
            "processed": False,
            "png_result_path": None,
            "tif_result_path": None,
        }
        image_name = os.path.basename(image["image_path"])
        logging.info(image["image_path"])
        img = tiff.imread(image["image_path"]).astype("uint8")
        bboxes_tensor = image["bboxes"]
        bboxes_tensor = torch.tensor(  # pylint:disable=E1101
            bboxes_tensor, device=predictor.device
        )
        bboxes_tensor = predictor.transform.apply_boxes_torch(
            bboxes_tensor, img.shape[:2]
        )
        points_tensor = image["points"]
        labels_tensor = image["labels"]
        if points_tensor is not None and labels_tensor is not None:
            points_tensor = np.asarray(points_tensor)
            labels_tensor = np.asarray(labels_tensor)
            points_tensor = torch.tensor(  # pylint:disable=E1101
                points_tensor, device=predictor.device
            )
            points_tensor = predictor.transform.apply_coords_torch(
                points_tensor, img.shape[:2]
            )
            labels_tensor = torch.tensor(  # pylint:disable=E1101
                labels_tensor, device=predictor.device
            )

        if bboxes_tensor is not None and points_tensor is not None:
            intersections = (
                (points_tensor >= bboxes_tensor[:, None, :2])
                & (points_tensor <= bboxes_tensor[:, None, 2:])
            ).all(2)
            list_points = []
            list_labels = []
            for i in range(bboxes_tensor.shape[0]):
                tmp_points = points_tensor[intersections[i, :]]
                tmp_labels = labels_tensor[intersections[i, :]]
                if i == 0:
                    min_point_box = tmp_points.shape[0]
                if tmp_points.shape[0] < min_point_box:
                    min_point_box = tmp_points.shape[0]
                list_points.append(tmp_points)
                list_labels.append(tmp_labels)

            points_tensor = np.zeros(
                (bboxes_tensor.shape[0], min_point_box, 2), dtype=np.float32
            )
            labels_tensor = np.zeros(
                (bboxes_tensor.shape[0], min_point_box), dtype=np.float32
            )
            for i in range(bboxes_tensor.shape[0]):
                points_tensor[i, :, :] = list_points[i][0:min_point_box]
                labels_tensor[i, :] = list_labels[i][0:min_point_box]
            points_tensor = torch.tensor(  # pylint:disable=E1101
                points_tensor, device=predictor.device
            )
            labels_tensor = torch.tensor(  # pylint:disable=E1101
                labels_tensor, device=predictor.device
            )

        predictor.set_image(img)
        masks, _, _ = predictor.predict_torch(
            point_coords=points_tensor,
            point_labels=labels_tensor,
            boxes=bboxes_tensor,
            multimask_output=False,
        )
        masks = np.array(masks)
        one_band_mask_1 = np.argmax(masks, axis=0)[0, ...]
        one_band_mask_2 = masks.sum(axis=0)[0, ...]
        one_band_mask = one_band_mask_1 + one_band_mask_2
        normalized_img = (one_band_mask - np.min(one_band_mask)) / (
            np.max(one_band_mask) - np.min(one_band_mask)
        )
        colored_img = plt.cm.tab20b(normalized_img)[:, :, :3]  # pylint: disable=E1101
        scaled_img = (colored_img * 255).astype(np.uint8)
        transparent_pixels = one_band_mask == 0
        scaled_img = np.concatenate(
            [
                scaled_img,
                255
                * np.ones(
                    (scaled_img.shape[0], scaled_img.shape[1], 1),
                    dtype=scaled_img.dtype,
                ),
            ],
            axis=2,
        )
        scaled_img[transparent_pixels, 3] = 0
        pil_image = Image.fromarray(scaled_img)
        mask_png_path = os.path.join(temp_path, image_name + "_bbox_mask.png")
        mask_tif_path = os.path.join(temp_path, image_name + "_bbox_mask.tif")
        pil_image.save(mask_png_path)
        save_numpy_as_geotiff(one_band_mask, image["image_path"], mask_tif_path)
        res_output = {
            "image_path": image["image_path"],
            "processed": True,
            "png_result_path": mask_png_path,
            "tif_result_path": mask_tif_path,
        }
        processing_status.append(res_output)
    return processing_status
