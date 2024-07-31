import paho.mqtt.client as mqtt
import yaml
from pathlib import Path


# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe("/camera_interface")  # Subscribe to the topic


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    print(f"Received message '{msg.payload.decode()}' on topic '{msg.topic}'")


if __name__ == "__main__":
    YAML_FILE_PATH = "CONFIG.yml"
    with Path(YAML_FILE_PATH).open() as f:
        CFG = yaml.load(f, Loader=yaml.Loader)

    # MQTT configuration
    BROKER_ADDRESS = CFG["MQTT_HOST"]
    PORT = CFG["MQTT_PORT"]

    # Create an MQTT client and attach our routines to it.
    mqtt_client = mqtt.Client()
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message

    mqtt_client.connect(BROKER_ADDRESS, PORT)

    # Blocking call that processes network traffic, dispatches callbacks and handles reconnecting.
    mqtt_client.loop_forever()
