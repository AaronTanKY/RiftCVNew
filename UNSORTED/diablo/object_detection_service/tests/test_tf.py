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
import tensorflow as tf

import numpy as np

from pathlib import Path

import base64

# import tensorflow_hub as hub
# ! REFACTOR THIS FILE


def image_from_bytes(img_byte):
    jpg_original = base64.b64decode(img_byte)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    frame = cv2.imdecode(jpg_as_np, flags=1)
    return frame


class MobileMOTModel:
    def __init__(self):
        # MODEL_PATH = "./models/lite-model_efficientdet_lite2_detection_default_1.tflite"
        MODEL_PATH = "./models/lite-model_efficientdet_lite0_int8_1.tflite"
        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    # TODO: BUG: WHY BBOX SO SMALL
    def infer_object_detection(self, request, context):
        a = request.image  # bytes
        jpg_original = base64.b64decode(a)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        frame = cv2.imdecode(jpg_as_np, flags=1)
        print(np.array(frame, dtype=np.uint8).shape)
        # Define new size
        NEW_WIDTH = 320
        NEW_HEIGHT = 320
        new_size = (NEW_WIDTH, NEW_HEIGHT)

        # Resize image
        resized_image = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        # print("Resized image 0:", resized_image)
        # resized_image = (resized_image / 255.0) * 255
        # print("Resized image 1:", resized_image)

        input_type = self.input_details[0]["dtype"]
        if input_type == np.uint8:
            input_scale, input_zero_point = self.input_details[0]["quantization"]
            print("Input scale:", input_scale)
            print("Input zero point:", input_zero_point)
        #     print()
        #     resized_image = (resized_image / input_scale) + input_zero_point
        #     resized_image = np.around(resized_image)

        #     print("Resized image 2:", resized_image)

        # ! TODO: I THINK RESIZING IS NOT THE BEST METHOD
        new_resized_image = resized_image.astype(np.uint8)
        # Add a batch dimension
        new_resized_image = np.expand_dims(new_resized_image, axis=0)
        print(new_resized_image)

        # detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite0/detection/1")
        # boxes, scores, classes, num_detections = detector(new_resized_image)

        # print("boxes", boxes)
        # print("scores", scores)
        # print("classes", classes)
        # boxes, scores, classes = boxes[0], scores[0], classes[0]

        # Set the tensor to point to the input data to be inferred.
        self.interpreter.set_tensor(self.input_details[0]["index"], new_resized_image)
        # Run the inference.
        self.interpreter.invoke()

        # If the output type is int8 (quantized model), rescale data
        output_type = self.output_details[0]["dtype"]
        print(self.output_details[0])
        print("Output Type: ", output_type)
        if output_type == np.uint8:
            output_scale, output_zero_point = self.output_details[0]["quantization"]
            print("Raw output scores:", output)
            print("Output scale:", output_scale)
            print("Output zero point:", output_zero_point)
            print()
            output = output_scale * (output.astype(np.float32) - output_zero_point)

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

        # ! TODO: FIX BUG OF NEGATIVE AND OVER POSTIVE VALUES OF BOUNDING BOX
        # At this point highest_score_box values are between 0 and 1 (proportional to image size)
        for index, scale_value in enumerate(highest_score_box):
            if scale_value < 0:
                scale_value = 0
            elif scale_value > 1:
                scale_value = 1
            highest_score_box[index] = scale_value

        WIDTH = 480
        HEIGHT = 640
        # 1 > 2
        highest_score_box[0] = highest_score_box[0] * NEW_WIDTH
        highest_score_box[1] = highest_score_box[1] * NEW_HEIGHT
        highest_score_box[2] = highest_score_box[2] * NEW_WIDTH
        highest_score_box[3] = highest_score_box[3] * NEW_HEIGHT

        # Now you have the class, bounding box, and score of the detection with the highest confidence
        if highest_score_class is not None:
            print("Class ID:", highest_score_class)
            print("Bounding Box:", highest_score_box)
            print("Confidence Score:", highest_score)
        else:
            print("No detections with significant confidence.")

        # ! WASTE OF COMMUNTATION
        _, buffer = cv2.imencode(".jpg", resized_image)
        data = base64.b64encode(buffer)

        if highest_score_class == 0:
            class_name = "person"
        else:
            class_name = "not person"

        detections = {
            "image": data,
            "detections": [
                {
                    # TODO: !USE CLASS NAME INSTEAD OF ID
                    "class_name": class_name,
                    # TODO: Confirm whether correct correlation
                    "x_low": highest_score_box[1],
                    "y_low": highest_score_box[0],
                    "x_high": highest_score_box[3],
                    "y_high": highest_score_box[2],
                    "confidence": highest_score,
                }
            ],
        }

        return pb2.ObjectDetectionReply(**detections)


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
