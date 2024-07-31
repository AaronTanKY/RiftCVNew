#!/usr/bin/env python
from dataclasses import dataclass
import cv2

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image 
from cv_bridge import CvBridge

from custom_interfaces.msg import NormalImageArray

class ObjectDetectionTokenSubscriber(Node):
    def __init__(self):
        BUFFER_SIZE = 10

        super().__init__('object_detection_token_subscriber')
        self.publisher_ = self.create_subscription(NormalImageArray, 'object_detection_token', self.print_images_in_array, BUFFER_SIZE) 
        
        self.bridge = CvBridge()

    def print_images_in_array(self, msg):
          for image in msg.image_array1:
              cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
              cv2.imshow("Image window", cv_image)
              cv2.waitKey(3)


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = ObjectDetectionTokenSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy

if __name__ == '__main__':
    main()