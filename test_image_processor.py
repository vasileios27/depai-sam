import grpc
import geopandas as gpd
import numpy as np
import pytest
import tifffile as tiff
import model_pb2
import model_pb2_grpc


def geographic_to_pixel_bbox(
    bbox_geo: np.array,
    image_width: int,
    image_height: int,
    min_latitude: float,
    max_latitude: float,
    min_longitude: float,
    max_longitude: float,
) -> np.array:
    lat_range = max_latitude - min_latitude
    lon_range = max_longitude - min_longitude
    x_min = ((bbox_geo[:, 0] - min_longitude) / lon_range * image_width).astype(int)
    y_min = ((max_latitude - bbox_geo[:, 3]) / lat_range * image_height).astype(int)
    x_max = ((bbox_geo[:, 2] - min_longitude) / lon_range * image_width).astype(int)
    y_max = ((max_latitude - bbox_geo[:, 1]) / lat_range * image_height).astype(int)
    pixel_bbox = np.column_stack((x_min, y_min, x_max, y_max))
    return pixel_bbox


def geometry_to_xy(gdf):
    list_bboxes = []
    for idx, row in gdf.iterrows():
        x_min, y_min, x_max, y_max = row.geometry.bounds
        list_bboxes.append([x_min, y_min, x_max, y_max])
    return list_bboxes


@pytest.fixture(scope="module")
def grpc_stub():
    channel = grpc.insecure_channel("localhost:8061")
    stub = model_pb2_grpc.ImageProcessorStub(channel)
    yield stub
    channel.close()


def test_process_image(grpc_stub):
    img_path = "./test-data/T40RBN_20230607T064629_RGB.tif"
    img = tiff.imread(img_path)
    roi_gdf = gpd.read_file("./test-data/palm_roi.shp")
    bbox_gdf = gpd.read_file("./test-data/bbox.shp")
    roi_bbox = geometry_to_xy(roi_gdf)[0]
    list_bboxes = geometry_to_xy(bbox_gdf)
    pixel_bboxes = geographic_to_pixel_bbox(
        np.array(list_bboxes),
        img.shape[1],
        img.shape[0],
        roi_bbox[1],
        roi_bbox[3],
        roi_bbox[0],
        roi_bbox[2],
    )
    pixel_bboxes = pixel_bboxes.tolist()
    data = [
        model_pb2.ImageRequest.Image(  # pylint:disable=E1101
            image_path=img_path,
            bboxes=[
                model_pb2.BoundingBox(  # pylint:disable=E1101
                    x_min=pixel_bboxes[0][0],
                    y_min=pixel_bboxes[0][1],
                    x_max=pixel_bboxes[0][2],
                    y_max=pixel_bboxes[0][3],
                )
            ],
            points=None,
            labels=None,
        )
    ]
    response = grpc_stub.ProcessImage(
        model_pb2.ImageRequest(images=data)  # pylint: disable=E1101
    )
    assert response.entries[0].processed is not None
    print("PNG Output File: " + response.entries[0].png_result_path)
    print("GTiff Output File: " + response.entries[0].tif_result_path)
