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
from numpy import random

import sys


class MobileMOTModel:
    def __init__(self):
        torch.backends.quantized.engine = "qnnpack"

        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
        cap.set(cv2.CAP_PROP_FPS, 36)

        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        net = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
        # jit model to take it from ~20fps to ~30fps
        net = torch.jit.script(net)

    def infer(self):
        with torch.no_grad():
            while True:
                # read frame
                ret, image = self.cap.read()
                if not ret:
                    raise RuntimeError("failed to read frame")

                # convert opencv output from BGR to RGB
                image = image[:, :, [2, 1, 0]]
                permuted = image

                # preprocess
                input_tensor = self.preprocess(image)

                # create a mini-batch as expected by the model
                input_batch = input_tensor.unsqueeze(0)

                # run model
                output = self.net(input_batch)

                return output


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
