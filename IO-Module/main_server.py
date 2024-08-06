from concurrent import futures
import logging

import grpc
import cv2

import base64

import multiprocessing as mp

from pathlib import Path

import numpy as np

import pyrealsense2 as rs

import subprocess
import multiprocessing

import yaml

import argparse
import sys

import os

import psutil
from google.protobuf.timestamp_pb2 import Timestamp
from datetime import datetime
import threading
import time

dir_path = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the current script
sys.path.append(os.path.join(dir_path, "protobufs"))
import main_server2_pb2 as pb2
import main_server2_pb2_grpc as pb2_grpc

sys.path.pop()

parser = argparse.ArgumentParser(description="Choosing between ROS1 or ROS2 Camera Interface")
parser.add_argument(
    "-c",
    "--camera_interface",
    type=int,
    default=3,
    help="Default value is 3 for MISB. You can set to 2 for ROS2 camera interface, 1 for ROS1",
)


args = parser.parse_args()

dir_path = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the current script
head, tail = os.path.split(dir_path)
sys.path.append(os.path.join(head, "Source-Module"))
if args.camera_interface == 1:
    from MPcam_ROS1 import Cam
elif args.camera_interface == 2:
    import rclpy
    from MPCam_ROS2 import Cam
elif args.camera_interface == 3:
    from MPCam_MISB5 import Cam
else:
    raise Exception("Unacceptable camera interface version")

sys.path.pop()


class Greeter(pb2_grpc.GreeterServicer):
    def __init__(self):
        if args.camera_interface in [1, 2]:
            self.cam = Cam()
            self._i = rs.intrinsics()
            self._depth = np.zeros((480, 640, 1), np.uint8)
        else:  # MISB
            self.cam = Cam("udp://239.0.0.1:1234")  # Replace with your MISB video source
            self._i = None  # MISB doesn't use RealSense intrinsics
            self._depth = None  # MISB typically doesn't have depth data
            self.cam.start()

    def get_rframe(self, request, context):  # asking for img
        if args.camera_interface == 2:  # ! Allows Swapping between ROS1 and ROS2 Camera interface
            rclpy.spin_once(self.cam)

        img = self.cam.get_frame()
        if img is not None:
            _, buffer = cv2.imencode(".jpg", img)
            data = base64.b64encode(buffer)
            return pb2.HelloReply(message=data)  # request.name) #'Hello, %s!' %
        else:
            return pb2.HelloReply(message=b'')
    
    def get_metadata(self, request, context):
        if args.camera_interface == 3:  # MISB
            metadata = self.cam.get_metadata()
            
            if metadata is not None:
                return pb2.MISBMetadata(metadataitem = metadata)
            else:
                return pb2.MISBMetadata(metadataitem=b"")
            
            # misb_metadata = pb2.MISBMetadata()
            # if metadata is None:
            #     print("Warning: Metadata is None. Returning empty metadata.")
            #     return misb_metadata
            
            # for key, value_tuple in metadata.items():
            #     item = pb2.MISBMetadata.MetadataItem(
            #         name=value_tuple[0],
            #         description=value_tuple[1],
            #         alternate_name=value_tuple[2]
            #     )
                
            #     # Handle different value types
            #     if isinstance(value_tuple[3], str):
            #         item.string_value = value_tuple[3]
            #     elif isinstance(value_tuple[3], (int, float)):
            #         item.double_value = float(value_tuple[3])
            #     elif isinstance(value_tuple[3], datetime):
            #         timestamp = Timestamp()
            #         timestamp.FromDatetime(value_tuple[3])
            #         item.timestamp_value.CopyFrom(timestamp)
            #     else:
            #         # For any other types, convert to string
            #         item.string_value = str(value_tuple[3])
                
            #     misb_metadata.items[key].CopyFrom(item)
            
            # return misb_metadata
        else:
            return pb2.MISBMetadata(metadataitem=b"")  # Empty for non-MISB

    def get_infrared_frame(self, request, context):  # asking for img
        if args.camera_interface in [1, 2]:
            if args.camera_interface == 2:
                rclpy.spin_once(self.cam)
            img = self.cam.get_infrared_frame()
            _, buffer = cv2.imencode(".jpg", img)
            data = base64.b64encode(buffer)
            return pb2.HelloReply(message=data)
        else:
            return pb2.HelloReply(message=b"")  # MISB doesn't have infrared

    def get_intrin(self, req, context):
        if args.camera_interface in [1, 2]:
            self._i = self.cam.get_intrinsics()
            coeffs = np.array(self._i.coeffs)
            c = base64.b64encode(coeffs)
            return pb2.intrin_list(
                coeffs=c, 
                width=self._i.width, 
                height=self._i.height,
                ppx=self._i.ppx, 
                ppy=self._i.ppy, 
                fx=self._i.fx, 
                fy=self._i.fy,
            )
        else:
            # For MISB, return metadata-based information if available
            frame = self.cam.get_frame()
            metadata = self.cam.get_metadata()  # Assume this method exists in MISB Cam
            return pb2.intrin_list(
                coeffs=b"",
                width=frame.shape[1] if frame is not None else 0,
                height=frame.shape[0] if frame is not None else 0,
                ppx = 0,
                ppy = 0,
                fx = 0,
                fy = 0


                #ppx=metadata.get('principal_point_x', 0) ,
                #ppy=metadata.get('principal_point_y', 0),
                #fx=metadata.get('focal_length_x', 0),
                #fy=metadata.get('focal_length_y', 0)
            )

    def get_depth(self, req, context):
        if args.camera_interface in [1, 2]:
            self._depth = self.cam.get_depth()
            depth_frame = self._depth.astype("uint16")
            d = base64.b16encode(depth_frame)
            return pb2.img(img_data=d)
        else:
            return pb2.img(img_data=b"")  # MISB typically doesn't have depth

    def ask_xyz(self, request, context):
        _x, _y, _z = self.cam.get_xyz(request.x, request.y)
        return pb2.xyz(x=str(_x), y=str(_y), z=str(_z))

    def ask_xyzmap(self, request, context):
        """Get the position of a detected object relative to the map frame.

        Args:
            request (data): protobuffer
            context (data): protobuffer

        Returns:
            data : protobuffer
        """
        new_xmap, new_ymap, new_zmap = self.cam.get_xyzmap(request.xnew, request.ynew, request.znew)
        return pb2.xyzmap(xnew=str(new_xmap), ynew=str(new_ymap), znew=str(new_zmap))

        # return pb2.xyzmap(xnew=str(3),ynew=str(3),znew=str(3))


def serve():
    yaml_source = os.path.join(os.path.dirname(__file__), "CONFIG.yml")
    with Path(yaml_source).open() as f:
        CFG = yaml.load(f, Loader=yaml.Loader)
        
    port = int(CFG["CAMERA_PORT"])
    kill_process_on_port(port)
    time.sleep(0.5)
    port = str(port)

    # Create greeter instance
    greeter = Greeter()

    # start the server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_GreeterServicer_to_server(greeter, server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()

# FOR WINDOWS ONLY 
def kill_process_on_port(port):
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            try:
                process = psutil.Process(conn.pid)
                process.terminate()
                print(f"Terminated process {conn.pid} using port {port}")
            except psutil.NoSuchProcess:
                pass

if __name__ == "__main__":
    logging.basicConfig()
    serve()

# channel = grpc.insecure_channel('localhost:50051') # Just for demonstration
# stub = pb2_grpc.GreeterStub(channel)
# request = # Code here to ask main_server for request, but should return pb2.xy(x="1.0", y="2.0")
# response = stub.ask_xyz(request)


