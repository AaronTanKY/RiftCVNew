import sys
import base64
import asyncio

# TODO: Future can add Multiprocessing: import torch.multiprocessing as mp
# import multiprocessing as mp
import cv2
import numpy as np
import ctypes
import time

import cv2
import numpy as np
import pyrealsense2 as rs

import rospy
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
import tf2_ros
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseStamped
import tf2_geometry_msgs
from geometry_msgs.msg import Pose

from MPCam import AbstractCam

# cam = jntest2.Camm()
# sleep(2)
# print("start")


# https://stackoverflow.com/questions/45311398/python-multiprocessing-class-methods
# watch -n 0.01 date +"%T.%3N"


class Cam(AbstractCam):
    def __init__(self):
        # threading.Thread(target=lambda: rospy.init_node("video_sub_py",disable_signals=True)).start() #disable signals
        
        # ! Use this if you are runnning cv on your laptop or gaming nuc
        rospy.Subscriber("/camera/color/image_raw", Image, self.load_color_frame)
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.load_depth_frame)
        rospy.Subscriber(
            "/camera/color/camera_info", CameraInfo, self.load_intrinsics
        )  # ? update rate / hertx
        rospy.Subscriber("/camera/infra1/image_rect_raw", Image, self.load_infrared_frame)

        # ! Use this if you are running with robomaster navigation
        # rospy.Subscriber("/d435i/color/image_raw", Image, self.load_color_frame)
        # rospy.Subscriber("/d435i/aligned_depth_to_color/image_raw", Image, self.load_depth_frame) #/camera/aligned_depth_to_color/image_raw (if this the shape isn't even the same)
        # rospy.Subscriber("/d435i/color/camera_info", CameraInfo, self.load_intrinsics) #update rate / hertx

        self._frame = np.zeros((480, 640, 3), np.uint8)
        self._depth_image = np.zeros((480, 640, 1), np.uint8)
        self._br = CvBridge()
        self._intrinsics = rs.intrinsics()

        self._pc = rs.pointcloud()

        self._imshow = True

        # jntest2
        rospy.init_node(
            "transform_test"
        )  # if have rospy.init_node, both threads must remove. If off rospy.init_node, 1 or 2 threads also can
        # threading.Thread(target=lambda: rospy.init_node("transform_test",disable_signals=True)).start()
        time.sleep(2)  # cannot remove, in seconds, only at the start
        # rospy.Subscriber("/camera/color/image_raw", Image, self.load_color_frame)
        rospy.Subscriber("/tf", TFMessage, self.load_transforms)
        self._tfBuffer = tf2_ros.Buffer()
        self._listener = tf2_ros.TransformListener(
            self._tfBuffer
        )  # need this if not cannot locate map frame in rviz
        self._pub = rospy.Publisher("posetest", PoseStamped, queue_size=1)

        self._data = TFMessage()

        print("init")

    def load_color_frame(self, ros_image):
        current_frame = self._br.imgmsg_to_cv2(ros_image)                   # Convert ROS image to cv2 readable format
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)      # Change color space (for whatever reason)

        self._frame = current_frame

    def load_infrared_frame(self, ros_image):
        current_frame = self._br.imgmsg_to_cv2(ros_image)
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

        self._infrared_frame = current_frame

    def load_depth_frame(self, data):
        d = self._br.imgmsg_to_cv2(data, data.encoding)  # convert to realsense frame failing
        if d is not None:
            self._depth_image = d  # int16
            # print(d.shape) #480. 848

    def load_intrinsics(self, cameraInfo):
        self._intrinsics.width = cameraInfo.width
        self._intrinsics.height = cameraInfo.height
        self._intrinsics.ppx = cameraInfo.K[2]
        self._intrinsics.ppy = cameraInfo.K[5]
        self._intrinsics.fx = cameraInfo.K[0]
        self._intrinsics.fy = cameraInfo.K[4]
        # self._intrinsics.model = cameraInfo.distortion_model
        self._intrinsics.model = rs.distortion.none  # dont need pass js set on the other side
        self._intrinsics.coeffs = [i for i in cameraInfo.D]

    def get_intrinsics(self):
        return super().get_intrinsics()

    def get_frame(self):
        return self._frame

    def get_infrared_frame(self):
        return self._infrared_frame

    def get_depth(self):
        return self._depth_image

    def get_xyz(self, x, y):  # most current frame, given xy, get z
        depth = self._depth_image[int(y)][int(x)]
        if depth != 0 and self._frame is not None and self._intrinsics is not None:
            result = rs.rs2_deproject_pixel_to_point(self._intrinsics, [int(x), int(y)], depth)
            print(result, "result")
            return (
                -result[0] / 1000,
                -result[1] / 1000,
                result[2] / 1000,
            )  # xyz POV of vehicle, result[0] [1] [2] is xyz POV of camera
        else:  # isnone
            return -1, -1, -1

    def get_xyzmap(self, x, y, z):  ## get xyz of target relative to map
        # print(type(x),y,z,'ffff')

        my_pose = Pose()
        my_pose.position.x = round(float(x))
        my_pose.position.y = round(float(y))
        my_pose.position.z = round(float(z))
        # same as axis
        my_pose.orientation.x = 0
        my_pose.orientation.y = 0
        my_pose.orientation.z = 0
        my_pose.orientation.w = 1
        print("test3")

        ##Use this if you are using the autonomous_exploration_development_environment file to run your rviz simulation
        xmap, ymap, zmap = self.transform_pose(
            my_pose, "vehicle", "map"
        )  # transform_pose( input_pose, from_frame, to_frame )
        # transform from "vehicle" frame to "map" frame

        ##Use this if you are running with robomaster navigation programme
        # xmap,ymap,zmap=self.transform_pose(my_pose,"camera_link","t265_odom_frame") #transform_pose( input_pose, from_frame, to_frame )
        # transform from "camera_link" frame to "t265_odom_frame" frame

        return xmap, ymap, zmap

    def get_crop(self, x_lo, y_lo, x_hi, y_hi):
        print("coords")
        print(x_lo, x_hi, y_lo, y_hi)
        cropped = self._frame[y_lo:y_hi, x_lo:x_hi]
        return cropped

    # jntest2
    def load_transforms(self, data):
        self._data = data

    def return_tf(self):
        return self._data

    def transform_pose(
        self, input_pose, from_frame, to_frame
    ):  # transform xyz relative to camera to xyz relative to starting point of robot
        # **Assuming /tf2 topic is being broadcasted
        # tf_buffer = tf2_ros.Buffer()
        # listener = tf2_ros.TransformListener(tf_buffer)

        pose_stamped = tf2_geometry_msgs.PoseStamped()
        pose_stamped.pose = input_pose
        pose_stamped.header.frame_id = from_frame
        pose_stamped.header.stamp = rospy.Time(0)  # can remove

        try:
            # ** It is important to wait for the listener to start listening. Hence the rospy.Duration(1)
            # take in vector and transform to map
            output_pose_stamped = self._tfBuffer.transform(
                pose_stamped, to_frame, timeout=rospy.Duration(2)
            )  # not the cause of lag
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            raise
        """
        output_pose_stamped.pose.position.x/=1000
        output_pose_stamped.pose.position.y/=1000
        output_pose_stamped.pose.position.z/=1000
        """

        print("HERRRRRRRRRRE" + output_pose_stamped.header.frame_id)
        # print('test5')

        self._pub.publish(output_pose_stamped)  # publish to ros topic posetest

        return (
            output_pose_stamped.pose.position.x,
            output_pose_stamped.pose.position.y,
            output_pose_stamped.pose.position.z,
        )  # PoseStamped()


"""
start_time=time.time()
end_time=time.time()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)
"""
