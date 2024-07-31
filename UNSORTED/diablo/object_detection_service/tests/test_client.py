import os
import json
import cv2
import grpc
import time

import main_server_pb2 as pb2
import main_server_pb2_grpc as pb2_grpc

import object_detection_pb2 as object_detection_pb2
import object_detection_pb2_grpc as object_detection_pb2_grpc

import base64
import numpy as np

import yaml
from pathlib import Path

import paho.mqtt.client as mqtt


def main():
    YAML_FILE_PATH = "CONFIG.yml"
    with Path(YAML_FILE_PATH).open() as f:
        CFG = yaml.load(f, Loader=yaml.Loader)

    stub = pb2_grpc.GreeterStub(grpc.insecure_channel("localhost:" + str(CFG["CAMERA_PORT"])))

    channel = grpc.insecure_channel(CFG["MOT_SERVICE_NAME"] + str(CFG["MOT_PORT"]))
    object_detection_stub = object_detection_pb2_grpc.objDetectorStub(channel)

    while True:
        response = stub.get_rframe(pb2.HelloRequest(name="y"))  # in bytes
        request = object_detection_pb2.ObjectDetectionImage(image=response.message)
        response = object_detection_stub.infer_object_detection(request)


if __name__ == "__main__":
    main()
