# TODO: MAKE THIS TEST FILE TO CONVENTION
import paho.mqtt.client as mqtt
import json
import base64
import numpy as np
import cv2


def on_message(client, userdata, msg):
    try:
        json_message = json.loads(msg.payload)
        print(json_message)
        list_of_target_chips = json_message["targetChips"]
        if list_of_target_chips == [None]:
            print("No Targets")
        else:
            base64_string = list_of_target_chips
    except Exception as e:
        print(f"Error: {e}")


# Create an MQTT client instance
client = mqtt.Client()

# Set the callback function
client.on_message = on_message

# Connect to the broker (replace with the address of your broker)
client.connect("localhost", 1883, 60)

# Subscribe to a topic (replace 'your/topic' with the desired topic)
client.subscribe("cerebro/gestalt/redForceReport")

# Start the client loop to keep the connection alive and listen for messages
client.loop_forever()
