import json
import grpc
import time

import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the current script
sys.path.append(os.path.join(dir_path, "../protobufs"))

import newMOT_pb2 as motor_pb2
import newMOT_pb2_grpc as motor_pb2_grpc

import main_server_pb2 as pb2
import main_server_pb2_grpc as pb2_grpc

sys.path.pop()

import yaml
from pathlib import Path

import paho.mqtt.client as mqtt

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from utils.image_convertor import image_from_bytes
from utils.image_convertor import convert_images_to_base64_str
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


def calculate_bearing(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Calculate the change in coordinates
    dlon = lon2 - lon1

    # Calculate the bearing
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - (sin(lat1) * cos(lat2) * cos(dlon))
    initial_bearing = atan2(x, y)

    # Convert bearing from radians to degrees and normalize
    initial_bearing = degrees(initial_bearing)
    bearing = (initial_bearing + 360) % 360

    return bearing


from math import asin, atan2, cos, degrees, radians, sin, atan, sqrt


def calculate_lat_long_at_distance(lat1, lon1, d, bearing, R=6371):
    """
    lat: initial latitude, in degrees
    lon: initial longitude, in degrees
    d: target distance from initial
    bearing: (true) heading in degrees
    R: optional radius of sphere, defaults to mean radius of earth

    Returns new lat/lon coordinate {d}km from initial, in degrees
    """
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    a = radians(bearing)
    lat2 = asin(sin(lat1) * cos(d / R) + cos(lat1) * sin(d / R) * cos(a))
    lon2 = lon1 + atan2(sin(a) * sin(d / R) * cos(lat1), cos(d / R) - sin(lat1) * sin(lat2))
    return (
        degrees(lat2),
        degrees(lon2),
    )


# TODO: Change name of class to reflect the output better
class ObjectDetectionRedForcePublisher(Node):
    def __init__(self, mqtt_client, camera_stub, mot_stub, initialized_message, quality, hz):
        super().__init__("object_detection_red_force_publisher")
        self.quality = quality
        self.hz = hz

        self.front_lat_long_subscriber = self.create_subscription(
            NavSatFix, "/gx5/gnss1/fix/relay", self.handle_front_lat_long_update, 10
        )
        self.backs_lat_long_subscriber = self.create_subscription(
            NavSatFix, "/gx5/gnss2/fix/relay", self.handle_back_lat_long_update, 10
        )

        self.TOPIC = "cerebro/gestalt/redForceReport"
        self.initialized_message = initialized_message
        self.mqtt_client = mqtt_client
        self.camera_stub = camera_stub
        self.mot_stub = mot_stub
        self.front_lat_long = {"lat": 0, "lng": 0}
        self.back_lat_long = {"lat": 0, "lng": 0}

    def get_relative_position_of_object(self, x_mid: float, y_mid: float):
        """Get the relative position of the object from the camera POV

        Args:
            x_mid (float): center point x coordinate
            y_mid (float): center point y coordinate

        Raises:
            Exception: Coordinates of the detected object is 0 0 0
            Exception: Coordinates of the detected object is -1 -1 -1

        Returns:
            x, y, z (float): x+ = Relative coordinates of the object from the camera POV: Right from center of camera, y+ up from center of camera.
        """
        coords = self.camera_stub.ask_xyz(
            pb2.xyz(x=str(round(x_mid)), y=str(round(y_mid)))
        )  # xyz POV of vehicle
        print(coords)
        if coords.z != (-1):  # its -1 -1 -1 if camera face table
            if coords.x != 0 and coords.y != 0 and coords.z != 0:
                x = float(coords.x)
                y = float(coords.y)
                z = float(coords.z)
                return x, y, z
            else:
                raise Exception("Coordinates of the detected object is 0 0 0")
        else:
            raise Exception("Coordinates of the detected object is -1 -1 -1")

    def calculate_xy_map_position_of_object(
        self, x_position_relative_to_robot, y_position_relative_to_robot
    ):
        relative_angle_object_from_robot_in_degrees = degrees(
            atan2(y_position_relative_to_robot, x_position_relative_to_robot)
        )

        distance_of_object_from_robot = sqrt(
            x_position_relative_to_robot**2 + y_position_relative_to_robot**2
        )

        get_bearing_of_robot_in_degrees = calculate_bearing(
            self.front_lat_long["lat"],
            self.front_lat_long["long"],
            self.back_lat_long["lat"],
            self.back_lat_long["long"],
        )

        bearing_of_object_in_degrees = (
            get_bearing_of_robot_in_degrees + relative_angle_object_from_robot_in_degrees
        )

        object_lat, object_long = calculate_lat_long_at_distance(
            lat1=self.front_lat_long["lat"],
            lon1=self.front_lat_long["long"],
            d=distance_of_object_from_robot,
            bearing=bearing_of_object_in_degrees,
        )

        return object_lat, object_long

    def handle_front_lat_long_update(self, msg):
        self.front_lat_long = {"lat": msg.latitude, "long": msg.longitude}
        self.try_publish()

    def handle_back_lat_long_update(self, msg):
        self.back_lat_long = {"lat": msg.latitude, "long": msg.longitude}
        self.try_publish()

    def try_publish(self):
        if self.front_lat_long is not None and self.back_lat_long is not None:
            self.publish()
            # ! This reseting function is the best thing I can think of to limit ensure newest lat long messages are used
            self.front_lat_long = None  # Reset after execution
            self.back_lat_long = None  # Reset after execution

    def publish(self):
        response = self.camera_stub.get_rframe(pb2.HelloRequest(name="y"))  # in bytes

        request = motor_pb2.MOT_pic(image=response.message)
        mot_results = self.mot_stub.askInfer_MOT(request)

        time_stamp_in_secs = time.time()
        date_time_iso_8601 = convert_seconds_to_date_time(time_stamp_in_secs)

        original_image = image_from_bytes(mot_results.image)

        detected_tokens_array = []
        relative_location_array = []
        map_location_array = []
        average_confidence = 0

        message = self.initialized_message  # TODO: Fix global variable

        for single_detection in mot_results.detections:
            x_low, y_low = int(single_detection.x_low), int(single_detection.y_low)
            x_high, y_high = int(single_detection.x_high), int(single_detection.y_high)

            if (
                x_low != x_high and y_low != y_high
            ):  # ?NOTE: Crude way to check whether detection exists
                cropped_image = original_image[y_low:y_high, x_low:x_high]

            x_mid = (x_high - x_low) / 2
            y_mid = (y_high - y_low) / 2

            if x_mid != 0 and y_mid != 0:  # if detect something
                # ? Get the relative position of the detected object
                (
                    x_position_object_from_camera,
                    y_position_object_from_camera,
                    z_position_object_from_camera,
                ) = self.get_relative_position_of_object(x_mid=x_mid, y_mid=y_mid)

                if (
                    x_position_object_from_camera != str(-1)
                    and y_position_object_from_camera != str(-1)
                    and z_position_object_from_camera != (-1)
                ):  # ? If error occured then default to relative x, y, z = 0
                    x_position_object_from_camera = 0
                    y_position_object_from_camera = 0
                    z_position_object_from_camera = 0

                # ? Convert to map coordinates
                # ! My own implementation of transform frame, can do better than this
                x_position_relative_to_robot = -x_position_object_from_camera
                y_position_relative_to_robot = z_position_object_from_camera
                z_position_object_from_camera = y_position_object_from_camera

                # ! Set altitude of object to 0
                z = 0
                x, y = self.calculate_xy_map_position_of_object(
                    x_position_relative_to_robot, y_position_relative_to_robot
                )

                relative_location_array.append(
                    {
                        "coordinate": [
                            x_position_relative_to_robot,
                            y_position_relative_to_robot,
                            z_position_object_from_camera,
                        ]
                    }
                )

                map_location_array.append(
                    [
                        x,
                        y,
                        z,
                    ]
                )

                if self.quality is not None:
                    base64_cropped_image_str = convert_images_to_base64_str(
                        cropped_image, self.quality
                    )

                # initialized_message.where.relativeLocation.coordinates.append([x, y, z])
                # ?data:image/jpg;base64 required for data processing by rift-proteus
                detected_tokens_array.append("data:image/jpg;base64," + base64_cropped_image_str)

                #!TODO: SHOULD WE USE AVERAGE CONFIDENCE
                average_confidence += single_detection.confidence

        if (
            len(detected_tokens_array) > 0
        ):  # ?NOTE: Best I can think for now to check whether detection exists
            average_confidence /= len(detected_tokens_array)
            message["targetChips"] = detected_tokens_array
            message["confidence"] = average_confidence
            message["messageHeader"]["timestamp"] = time_stamp_in_secs  # In seconds
            message["when"]["eventStart"] = date_time_iso_8601  #  Represented in ISO-8601 format

            message["where"]["relativeLocation"]["coordinates"] = relative_location_array
            message["where"]["location"]["coordinates"] = map_location_array

        encoded_message = encode_message_data(message)
        self.mqtt_client.publish(self.TOPIC, encoded_message)

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


def publish_start_loop(camera_stub, mot_stub, client, initialized_message, quality: int, hz):
    rclpy.init(args=None)
    publisher = ObjectDetectionRedForcePublisher(
        client, camera_stub, mot_stub, initialized_message, quality, hz
    )
    rclpy.spin(publisher)
    publisher.destroy_node()
    rclpy.shutdown()


def initialize_robot_data_in_schema(json_template_message, cerebro_id, detection_sensor) -> dict:
    json_template_message["cerebroId"] = cerebro_id
    json_template_message["detection"]["sensor"] = detection_sensor

    return json_template_message


def main():
    YAML_FILE_PATH = "../CONFIG.yml"
    with Path(YAML_FILE_PATH).open() as f:
        CFG = yaml.load(f, Loader=yaml.Loader)

    args = get_user_args()

    CEREBRO_ID = CFG["CEREBRO_ID"]
    SENSOR_TYPE = CFG["SENSOR_TYPE"]

    with open(
        "../cerebro_messages/cerebroRedForceReportMessageTemplate.json", "r"
    ) as json_template_file:
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
    mot_stub = motor_pb2_grpc.MOTorStub(channel)

    publish_start_loop(
        camera_stub, mot_stub, client, initialized_message, args.published_image_quality, args.hertz
    )


if __name__ == "__main__":
    main()
