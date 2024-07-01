import json
import sys

sys.path.append("./src")
from concurrent import futures
import time
import logging
import grpc
import model_pb2
import model_pb2_grpc
from app.inference import segment_sam_prompt

logging.getLogger().setLevel(logging.INFO)


class ImageProcessorServicer(model_pb2_grpc.ImageProcessorServicer):
    def ProcessImage(self, request, context):
        list_images = []
        list_images = []
        for image in request.images:
            bboxes = [
                [bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max]
                for bbox in image.bboxes
            ]
            points = json.loads(image.points.value) if image.points.value else None
            labels = json.loads(image.labels.value) if image.labels.value else None

            image_dict = {
                "image_path": image.image_path,
                "bboxes": bboxes,
                "points": points,
                "labels": labels,
            }
            list_images.append(image_dict)
        print(list_images)
        result = self.run_model(list_images)
        return model_pb2.ImageResponse(entries=result)  # pylint: disable=E1101

    def run_model(self, list_images):
        result = segment_sam_prompt(list_images)
        return result


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_pb2_grpc.add_ImageProcessorServicer_to_server(
        ImageProcessorServicer(), server
    )
    server.add_insecure_port("[::]:8061")
    server.start()
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve()
