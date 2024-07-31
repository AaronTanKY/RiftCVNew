import sys
import base64
import asyncio

# import multiprocessing as mp
import cv2
import numpy as np
import ctypes
import time


import rospy
import threading
from flask import Flask, Response
from std_msgs.msg import UInt32
from sensor_msgs.msg import PointCloud2

import cv2
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
import pyrealsense2 as rs

import sensor_msgs.point_cloud2 as pc2

import time

# import tf2_msgs
import tf2_ros
from tf2_msgs.msg import TFMessage
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

# https://stackoverflow.com/questions/45311398/python-multiprocessing-class-methods
# watch -n 0.01 date +"%T.%3N"


class Cam:
    def __init__(self):
        rospy.init_node("transform_test")
        # threading.Thread(target=lambda: rospy.init_node("pub_py",disable_signals=True)).start() #disable signals

        # rospy.Subscriber("/camera/color/image_raw", Image, self.load_color_frame)
        rospy.Subscriber("/tf", TFMessage, self.load_transforms)
        self._tfBuffer = tf2_ros.Buffer()
        self._listener = tf2_ros.TransformListener(self._tfBuffer)
        self._pub = rospy.Publisher("posetest", PoseStamped, queue_size=1)

        self._data = TFMessage()
        print("init")

    def load_transforms(self, data):
        self._data = data

    def return_tf(self):
        return self._data

    def pub_xyz(self, x, y, z):
        print("testtt")
        # rospy.init_node('nodelistener')
        try:
            print("test2")
            (trans, rot) = self._tfBuffer.lookup_transform(
                "vehicle", "camera", rospy.Time(0), rospy.Duration(1)
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return  # die

        print(type(rot))
        print(type(trans))

        p = PoseStamped()
        p.header.seq = 1
        p.header.stamp = rospy.Time.now()
        p.header.frame_id = "testpoint"

        p.pose.position.x = x
        p.pose.position.y = y
        p.pose.position.z = z

        # orient in same direction as robot+
        p.pose.orientation.x = self._data.transforms[0].transform.rotation.x
        p.pose.orientation.y = self._data.transforms[0].transform.rotation.y
        p.pose.orientation.z = self._data.transforms[0].transform.rotation.z
        p.pose.orientation.w = self._data.transforms[0].transform.rotation.w

        print(p, "test1")
        self._pub.publish(p)


a = Cam()
a.pub_xyz(1, 2, 3)
