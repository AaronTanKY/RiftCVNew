from abc import ABC, abstractmethod
import pyrealsense2 as rs


class AbstractCam:
    """
    Abstract class for camera interface
    """

    def __init__(self):
        self._intrinsics = rs.intrinsics()
        pass

    @abstractmethod
    def load_color_frame(self, data):
        pass

    @abstractmethod
    def load_depth_frame(self, data):
        pass

    @abstractmethod
    def load_intrinsics(self, camera_info):
        pass

    @abstractmethod
    def transform_pose(self, input_pose, from_frame, to_frame):
        pass

    @abstractmethod
    def get_xyz(self, x, y):
        pass

    @abstractmethod
    def get_xyzmap(self, x, y, z):
        pass

    @abstractmethod
    def get_crop(self, x_lo, y_lo, x_hi, y_hi):
        pass

    @abstractmethod
    def get_frame(self):
        pass

    @abstractmethod
    def get_infrared_frame(self):
        pass

    @abstractmethod
    def get_depth(self):
        pass

    def get_intrinsics(self):
        return self._intrinsics
