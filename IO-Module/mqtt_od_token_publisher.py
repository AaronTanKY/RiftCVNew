import os
import json
import cv2
import grpc
import time
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the current script
sys.path.append(os.path.join(dir_path, "protobufs"))

import object_detection_pb2 as object_detection_pb2
import object_detection_pb2_grpc as object_detection_pb2_grpc

import main_server_pb2 as pb2
import main_server_pb2_grpc as pb2_grpc

sys.path.pop()

import base64
import numpy as np

import yaml
from pathlib import Path

import paho.mqtt.client as mqtt


def image_from_bytes(img_byte):
    jpg_original = base64.b64decode(img_byte)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    frame = cv2.imdecode(jpg_as_np, flags=1)
    return frame


def convert_images_to_base64_str(image, quality=90) -> str:
    assert (
        isinstance(quality, int) and 0 <= quality <= 100
    ), "Quality value must be an integer between 0 and 100."

    _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    base64_image_str = base64.b64encode(buffer).decode("utf-8")
    return base64_image_str


from datetime import datetime, timezone


def convert_seconds_to_date_time(time_stamp_in_secs) -> datetime:
    """Converts seconds to date time represented in ISO-8601 format

    Args:
        time_stamp_in_secs (float): time since UTC epoch in seconds

    Returns:
        date_time: ISO-8601 format
    """
    # Convert the timestamp to a timezone-aware datetime object in UTC
    dt_utc = datetime.fromtimestamp(time_stamp_in_secs, timezone.utc)

    # Format the datetime object to the desired format
    formatted_time = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    return formatted_time


def encode_message_data(message):
    # TODO: Best within the time provided I chose JSON encoding, but I think better solutions are out there
    encoded_message = json.dumps(message)
    return encoded_message


class ObjectDetectionRedForcePublisher:
    def __init__(
        self, mqtt_client, camera_stub, object_detection_stub, initialized_message, quality, hz
    ):
        self.quality = quality
        self.hz = hz

        self.TOPIC = "cerebro/gestalt/redForceReport"
        self.initialized_message = initialized_message
        self.mqtt_client = mqtt_client
        self.camera_stub = camera_stub
        self.object_detection_stub = object_detection_stub

    def publish(self):
        response = self.camera_stub.get_rframe(pb2.HelloRequest(name="y"))  # in bytes

        request = object_detection_pb2.ObjectDetectionImage(image=response.message)
        object_detection_results = self.object_detection_stub.infer_object_detection(request)

        time_stamp_in_secs = time.time()
        date_time_iso_8601 = convert_seconds_to_date_time(time_stamp_in_secs)

        original_image = image_from_bytes(object_detection_results.image)

        message = self.initialized_message  # TODO: Fix global variable

        single_detection = object_detection_results.detections[0]

        x_low, y_low = int(single_detection.x_low), int(single_detection.y_low)
        x_high, y_high = int(single_detection.x_high), int(single_detection.y_high)

        if (
            x_low != x_high and y_low != y_high
        ):  # ?NOTE: Crude way to check whether detection exists
            cropped_image = original_image[y_low:y_high, x_low:x_high]

            base64_cropped_image_str = convert_images_to_base64_str(cropped_image, self.quality)

            if single_detection.class_name == "person":
                # ?data:image/jpg;base64 required for data processing by rift-proteus
                message["targetChips"] = "data:image/jpg;base64," + base64_cropped_image_str
                message["confidence"] = single_detection.confidence
                message["messageHeader"]["timestamp"] = time_stamp_in_secs  # In seconds
                message["when"][
                    "eventStart"
                ] = date_time_iso_8601  #  Represented in ISO-8601 format

        encoded_message = encode_message_data(message)
        self.mqtt_client.publish(self.TOPIC, encoded_message)

        # ! TODO: Find better way to put upper limit to hertz
        time.sleep(1 / self.hz)


class MQTTClient:
    def __init__(self, broker, port, keepalive=60):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_publish = self.on_publish
        self.client.connect(broker, port, keepalive)

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to broker")
        else:
            print("Connection failed")

    def on_publish(self, client, userdata, mid):
        print("Message published")

    def start_client(self):
        self.client.loop_start()  # Start the MQTT loop in the background

    def get_client(self):
        return self.client

    def publish(self, topic, message):
        self.client.publish(topic, message)


def initialize_robot_data_in_schema(json_template_message, cerebro_id, detection_sensor) -> dict:
    json_template_message["cerebroId"] = cerebro_id
    json_template_message["detection"]["sensor"] = detection_sensor

    return json_template_message


def publish_start_loop(camera_stub, mot_stub, client, initialized_message, quality: int, hz):
    publisher = ObjectDetectionRedForcePublisher(
        client, camera_stub, mot_stub, initialized_message, quality, hz
    )

    # TODO: Must I do clean up?
    while True:
        publisher.publish()


import argparse


def get_user_args():
    parser = argparse.ArgumentParser(description="Inputs to the publisher")
    parser.add_argument(
        "-q",
        "--published_image_quality",
        type=int,
        default=90,
        help="Quality of published image. Note: Only image in base64 string representation is published",
    )
    parser.add_argument(
        "--hertz",
        type=float,
        default=0.5,
        help="Hz of published message",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    YAML_FILE_PATH = os.path.join(os.path.dirname(__file__), "CONFIG.yml")

    with Path(YAML_FILE_PATH).open() as f:
        CFG = yaml.load(f, Loader=yaml.Loader)

    args = get_user_args()

    CEREBRO_ID = CFG["CEREBRO_ID"]
    SENSOR_TYPE = CFG["SENSOR_TYPE"]

    CEREBRO_FILE_PATH = os.path.join(os.path.dirname(__file__), "cerebro_messages/cerebroRedForceReportMessageTemplate.json")
    with open(CEREBRO_FILE_PATH, "r") as json_template_file:
        json_template_message = json.load(json_template_file)

    initialized_message = initialize_robot_data_in_schema(
        json_template_message, CEREBRO_ID, SENSOR_TYPE
    )

    BROKER_ADDRESS = CFG["MQTT_HOST"]
    PORT = CFG["MQTT_PORT"]
    client = MQTTClient(BROKER_ADDRESS, PORT)
    client.start_client()

    camera_stub = pb2_grpc.GreeterStub(
        grpc.insecure_channel("localhost:" + str(CFG["CAMERA_PORT"]))
    )

    channel = grpc.insecure_channel(
        "localhost:" + str(CFG["MOT_PORT"])
    )  # TODO: Set back to mot_service
    mot_stub = object_detection_pb2_grpc.objDetectorStub(channel)

    publish_start_loop(
        camera_stub, mot_stub, client, initialized_message, args.published_image_quality, args.hertz
    )
