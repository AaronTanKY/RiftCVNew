import paho.mqtt.client as mqtt
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import yaml
from pathlib import Path


class MqttPublisher(Node):
    def __init__(self, client):
        self.TOPIC = "/chatter"
        super().__init__("mqtt_publisher")

        self.ros_subscription = self.create_subscription(String, self.TOPIC, self.ros_callback)
        
    def ros_callback(self, msg):
        self.mqtt_client.publish(self.TOPIC, msg.data)


if __name__ == "__main__":
    YAML_FILE_PATH = "CONFIG.yml"
    with Path(YAML_FILE_PATH).open() as f:
        CFG = yaml.load(f, Loader=yaml.Loader)

    rclpy.init()

    BROKER_ADDRESS = CFG["MQTT_HOST"]
    PORT = CFG["MQTT_PORT"]
    mqtt_client = mqtt.Client()
    mqtt_client.connect(BROKER_ADDRESS, PORT)

    node = MqttPublisher(mqtt_client)
    rclpy.spin(node)
    rclpy.shutdown()
