from __future__ import print_function
import time

import numpy as np

import grpc
import newMOT_pb2 as pb2
import newMOT_pb2_grpc as pb2_grpc

import cv2
from PIL import Image
import subprocess
import yaml
from concurrent import futures
import logging

import grpc
import tensorflow as tf

import cv2
import numpy as np

from pathlib import Path

import torch

import base64


def image_from_bytes(img_byte):
    jpg_original = base64.b64decode(img_byte)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    frame = cv2.imdecode(jpg_as_np, flags=1)
    return frame


import onnx
import onnxruntime as ort
import numpy as np


class MobileMOTModel:
    def __init__(self):
        MODEL_PATH = "lite-model_ssd_mobilenet_v2_100_int8_default_1.tflite"
        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        print(self.input_details)
        self.input_details[0]["shape"] = np.array([1, 360, 360, 3], dtype=np.int32)

    def askInfer_MOT(self, request, context):
        a = request.image  # bytes
        jpg_original = base64.b64decode(a)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        frame = cv2.imdecode(jpg_as_np, flags=1)

        # Define new size
        new_width = 360
        new_height = 360
        new_size = (new_width, new_height)

        # Resize image
        resized_image = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)

        # Normalize the image to be in the range [-128, 127]
        resized_image = ((resized_image / 255.0) * 255) - 128
        resized_image = resized_image.astype(np.int8)

        # Add a batch dimension
        image = np.expand_dims(resized_image, axis=0)

        # print(numpy_image.shape)
        # self.input_details[0]["shape"] = np.array([1, -1, -1, 3], dtype=np.int32)

        # Set the tensor to point to the input data to be inferred.
        self.interpreter.set_tensor(self.input_details[0]["shape"], image)

        # Run the inference.
        self.interpreter.invoke()

        # Extract the output data.
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])

        # output_dict = {
        #     "num_detections": int(self.interpreter.get_tensor(self.outputs[3]["index"])),
        #     "detection_classes": self.interpreter.get_tensor(self.outputs[1]["index"]).astype(
        #         np.uint8
        #     ),
        #     "detection_boxes": self.interpreter.get_tensor(self.outputs[0]["index"]),
        #     "detection_scores": self.interpreter.get_tensor(self.outputs[2]["index"]),
        # }
        # print(output_dict)


def serve(port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_MOTorServicer_to_server(MobileMOTModel(), server)
    server.add_insecure_port("[::]:" + str(port))
    server.start()
    print("Server started, listening on " + str(port))
    server.wait_for_termination()


if __name__ == "__main__":
    with Path("CONFIG.yml").open() as f:
        CFG = yaml.load(f, Loader=yaml.Loader)
    cmd = "fuser -k " + str(CFG["MOT_PORT"]) + "/tcp"
    print(cmd)
    subprocess.run(cmd, shell=True)
    time.sleep(0.3)

    logging.basicConfig()
    serve(CFG["MOT_PORT"])
