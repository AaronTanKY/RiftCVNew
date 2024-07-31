import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import Image, CameraInfo
from tf2_ros import (
    Buffer,
    TransformListener,
    LookupException,
    ConnectivityException,
    ExtrapolationException,
)
from geometry_msgs.msg import PoseStamped
from tf2_geometry_msgs import (
    PoseStamped as TF2PoseStamped,
)  # This may need verification as the tf2_geometry_msgs package in ROS2 may differ
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
import cv2
import numpy as np
import pyrealsense2 as rs

from MPCam import AbstractCam


class Cam(Node, AbstractCam):
    def __init__(self):
        super().__init__("cam_node")

        self._frame = np.zeros((480, 640, 3), np.uint8)
        self._depth_image = np.zeros((480, 640), dtype=np.uint16)  # Depth image is uint16
        self._br = CvBridge()
        self._intrinsics = rs.intrinsics()
        self._pc = rs.pointcloud()
        self._imshow = True

        self.subscription_color = self.create_subscription(
            Image, "/camera/color/image_raw", self.load_color_frame, 10
        )
        self.subscription_depth = self.create_subscription(
            Image, "/camera/aligned_depth_to_color/image_raw", self.load_depth_frame, 10
        )
        self.subscription_camera_info = self.create_subscription(
            CameraInfo, "/camera/color/camera_info", self.load_intrinsics, 10
        )

        self.subscription_camera_info = self.create_subscsription(
            Image, "/camera/", self.load_intrinsics, 10
        )

        self._tfBuffer = Buffer()
        self._listener = TransformListener(self._tfBuffer, self)
        self._pub = self.create_publisher(PoseStamped, "posetest", 10)

        self.get_logger().info("Camera node initialized")

    def load_color_frame(self, data):
        current_frame = self._br.imgmsg_to_cv2(data, desired_encoding="bgr8")
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        self._frame = current_frame

    def load_depth_frame(self, data):
        print(data)
        d = self._br.imgmsg_to_cv2(data, desired_encoding=data.encoding)
        if d is not None:
            self._depth_image = d

    def load_intrinsics(self, camera_info):
        self._intrinsics.width = camera_info.width
        self._intrinsics.height = camera_info.height
        self._intrinsics.ppx = camera_info.k[2]
        self._intrinsics.ppy = camera_info.k[5]
        self._intrinsics.fx = camera_info.k[0]
        self._intrinsics.fy = camera_info.k[4]
        self._intrinsics.model = rs.distortion.none
        self._intrinsics.coeffs = list(camera_info.d)

    def transform_pose(self, input_pose, from_frame, to_frame):
        pose_stamped = PoseStamped()
        pose_stamped.pose = input_pose
        pose_stamped.header.frame_id = from_frame
        pose_stamped.header.stamp = self.get_clock().now().to_msg()

        try:
            transformed_pose = self._tfBuffer.transform(
                pose_stamped, to_frame, timeout=Duration(seconds=1)
            )
        except Exception as e:
            self.get_logger().error("Failed to transform pose: " + str(e))
            return None, None, None

        self._pub.publish(transformed_pose)

        return (
            transformed_pose.pose.position.x,
            transformed_pose.pose.position.y,
            transformed_pose.pose.position.z,
        )

    def get_xyz(self, x, y):
        depth = self._depth_image[int(y)][int(x)]
        self.get_logger().info(f"{depth} depth")
        if depth != 0 and self._frame is not None and self._intrinsics is not None:
            result = rs.rs2_deproject_pixel_to_point(self._intrinsics, [int(x), int(y)], depth)
            return result[2] / 1000, -result[0] / 1000, -result[1] / 1000
        else:
            return -1, -1, -1

    def get_xyzmap(self, x, y, z):
        my_pose = Pose()
        my_pose.position.x = round(float(x), 4)
        my_pose.position.y = round(float(y), 4)
        my_pose.position.z = round(float(z), 4)
        my_pose.orientation.w = 1

        try:
            pose_stamped = TF2PoseStamped()
            pose_stamped.pose = my_pose
            pose_stamped.header.frame_id = "vehicle"
            pose_stamped.header.stamp = self.get_clock().now().to_msg()

            transformed_pose = self._tfBuffer.transform(
                pose_stamped, "map", timeout=Duration(seconds=2)
            )

            self.publisher_pose.publish(transformed_pose)

            return (
                transformed_pose.pose.position.x,
                transformed_pose.pose.position.y,
                transformed_pose.pose.position.z,
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().error(f"Exception during transform: {e}")
            return None, None, None

    def get_crop(self, x_lo, y_lo, x_hi, y_hi):
        self.get_logger().info(f"coords: {x_lo}, {x_hi}, {y_lo}, {y_hi}")
        return self._frame[y_lo:y_hi, x_lo:x_hi]

    def get_frame(self):
        return self._frame

    def get_depth(self):
        return self._depth_image

    def get_intrinsics(self):
        return super().get_intrinsics()


rclpy.init(args=None)
