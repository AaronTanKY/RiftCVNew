from __future__ import print_function
import time

import torch
import numpy as np
from torchvision import models, transforms

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


import cv2
import numpy as np

from pathlib import Path

import torch
import json
import base64

from PIL import Image

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)


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
        model_path = "yolov7-tiny.onnx"

        # Set up the inference session
        self.ort_session = ort.InferenceSession(model_path)

        # Get the input name
        self.input_name = self.ort_session.get_inputs()[0].name

        self.transform = transforms.Compose(
            [
                transforms.Resize((640, 640)),  # Resize the image to 640x640
                transforms.ToTensor(),  # Convert the image to a PyTorch tensor
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet stats
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # torch.backends.quantized.engine = "qnnpack"

        # self.preprocess = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     ]
        # )

        # # model = models.quantization.mobilenet_v3_large(pretrained=True, quantize=True)
        # # # jit model to take it from ~20fps to ~30fps
        # # model = torch.jit.script(model)

        # # Step 1: Initialize model with the best available weights
        # weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        # model = fasterrcnn_resnet50_fpn_v2(weights=weights)

        # # set quantization config for server (x86)
        # BACKEND = "qnnpack"
        # model.qconfig = torch.quantization.get_default_qconfig(BACKEND)

        # # insert observers
        # torch.quantization.prepare(model, inplace=True)

        # # Calibrate the model and collect statistics
        # with torch.inference_mode():
        #     for _ in range(10):
        #         x = torch.rand(1, 2, 28, 28)
        #         model(x)

        # # convert to quantized version

        # torch.quantization.convert(model, inplace=True)

        # model.eval()
        # # self.preprocess = weights.transforms()
        # # self.model = model

        # print(model[[1]].weight().element_size())  # 1 byte instead of 4 bytes for FP32

    def askInfer_MOT(self, request, context):
        # ! Input is in shape [0, 3, 640, 640]
        # Define the necessary transformations
        a = request.image  # bytes
        jpg_original = base64.b64decode(a)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        frame = cv2.imdecode(jpg_as_np, flags=1)
        print(frame[0])
        image = Image.fromarray(frame)

        # Apply transformations
        image = self.transform(image)

        # Add a batch dimension by using 'unsqueeze'
        input_tensor = image.unsqueeze(0)  # Create a mini-batch as expected by the mode
        input_data = input_tensor.numpy()

        # Run the model
        outputs = self.ort_session.run(None, {self.input_name: input_data})
        print(outputs)
        return
        # class_probabilities = outputs[0][0][0][5:]
        # class_id = np.argmax(class_probabilities)
        # print(class_id)

        pass

        # classes = json.load(open("imagenet_class_index.json"))
        # with torch.no_grad():
        #     while True:
        #         a = request.image  # bytes
        #         jpg_original = base64.b64decode(a)
        #         jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        #         frame = cv2.imdecode(jpg_as_np, flags=1)
        #         image = Image.fromarray(frame)

        #         # Step 3: Apply inference preprocessing transforms
        #         batch = self.preprocess(image).unsqueeze(0)
        #         # Step 4: Use the model and print the predicted category
        #         prediction = self.model(batch).squeeze(0).softmax(0)
        #         class_id = prediction.argmax().item()
        #         score = prediction[class_id].item()
        #         category_name = weights.meta["categories"][class_id]
        #         print(f"{category_name}: {100 * score:.1f}%")
        #         print(weights.meta["categories"])

        #         return "dwa"

        # # preprocess
        # input_tensor = self.preprocess(frame)

        # # create a mini-batch as expected by the model
        # input_batch = input_tensor.unsqueeze(0)

        # # run model
        # output = self.net(input_batch)

        # top = list(enumerate(output[0].softmax(dim=0)))
        # top.sort(key=lambda x: x[1], reverse=True)

        # for idx, val in top[:10]:
        #     print(f"{val.item()*100:.2f}% {idx}")

        # return output


# @dataclass
# class SingleMOTDetectionResults:
#     frame_no: int
#     id: int
#     x_low: float
#     x_high: float
#     y_low: float
#     y_high: float
#     class_: int
#     confidence: int


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
