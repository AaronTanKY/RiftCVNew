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
from matplotlib import pyplot as plt


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
        MODEL_PATH = "lite-model_efficientdet_lite0_int8_1.tflite"
        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def askInfer_MOT(self, request, context):
        a = request.image  # bytes
        jpg_original = base64.b64decode(a)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        frame = cv2.imdecode(jpg_as_np, flags=1)

        # Define new size
        new_width = 320
        new_height = 320
        new_size = (new_width, new_height)

        # Resize image
        resized_image = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
        # cv2.imshow("l", np.array(frame, dtype=np.uint8))
        # cv2.waitKey(1)
        # Normalize the image to be in the range [-128, 127]
        resized_image = ((resized_image / 255.0) * 255) - 128

        # if input_type == np.uint8:
        #     input_scale, input_zero_point = self.input_details[0]["quantization"]
        #     print("Input scale:", input_scale)
        #     print("Input zero point:", input_zero_point)
        #     print()
        #     resized_image = (resized_image / input_scale) + input_zero_point
        #     resized_image = np.around(resized_image)

        resized_image = resized_image.astype(np.uint8)
        # Add a batch dimension
        resized_image = np.expand_dims(resized_image, axis=0)
        print(resized_image)

        # Set the tensor to point to the input data to be inferred.
        self.interpreter.set_tensor(self.input_details[0]["index"], resized_image)

        # Run the inference.
        self.interpreter.invoke()

        # Extract the output data.
        output = self.interpreter.get_tensor(self.output_details[0]["index"])

        boxes = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]["index"])[0]
        highest_score = 0

        highest_score_class = None
        highest_score_box = None

        # Iterate through all the scores
        for i in range(len(scores)):
            if scores[i] > highest_score:
                highest_score = scores[i]
                highest_score_class = int(classes[i])
                highest_score_box = boxes[i]

        # Now you have the class, bounding box, and score of the detection with the highest confidence
        if highest_score_class is not None:
            print("Class ID:", highest_score_class)
            print("Bounding Box:", highest_score_box)
            print("Confidence Score:", highest_score)
        else:
            print("No detections with significant confidence.")


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
