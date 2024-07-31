from __future__ import print_function
import time

import numpy as np

import grpc
import object_detection_pb2 as pb2
import object_detection_pb2_grpc as pb2_grpc

import cv2
import subprocess
import yaml
from concurrent import futures
import logging

import grpc
import cv2
import numpy as np

from pathlib import Path
import onnxruntime as rt

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
        MODEL_PATH = "yolov7_qat_640.onnx"
        onnx_model = onnx.load(MODEL_PATH)
        onnx.checker.check_model(onnx_model)

        # Print the model's input and output information
        print("Model Inputs:")
        for input in onnx_model.graph.input:
            print(input.name, input.type.tensor_type)

        print("\nModel Outputs:")
        for output in onnx_model.graph.output:
            print(output.name, output.type.tensor_type)

        self.ort_sess = ort.InferenceSession(MODEL_PATH)

        self.input_name = self.ort_sess.get_inputs()[0].name
        self.output_name = self.ort_sess.get_outputs()[0].name

    def infer_object_detection(self, request, context):
        a = request.image  # bytes
        jpg_original = base64.b64decode(a)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        frame = cv2.imdecode(jpg_as_np, flags=1)

        # ? This adding border method might not be the best way to do it
        left, right = 0, 0
        top = int((640 - 480) / 2)
        bottom = top
        borderType = cv2.BORDER_CONSTANT
        image = cv2.copyMakeBorder(frame, top, bottom, left, right, borderType)

        image = image.astype(np.uint8)
        transposed_image = np.transpose(image, (2, 0, 1))
        resized_image = np.expand_dims(transposed_image, axis=0)
        print(resized_image.shape)

        outputs = self.ort_sess.run([self.output_name], {"images": resized_image})

        for batch in outputs[0]:
            for detection in batch:
                # Extract information from each detection
                # Example: bbox coordinates, class ID, confidence score
                bbox = detection[:4]  # Placeholder: assuming first 4 values are bbox coordinates
                class_id = detection[4]  # Placeholder
                confidence = detection[5]  # Placeholder

                # Process the detection (e.g., filter based on confidence, map class ID to name)
                if confidence > 0.5:  # Define a suitable threshold
                    print(f"Class ID: {class_id}, Confidence: {confidence}, BBox: {bbox}")


def serve(port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_objDetectorServicer_to_server(MobileMOTModel(), server)
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
