import cv2
import numpy as np
from MPCam import AbstractCam
import klvdata
from klvdata import misb0601 
import subprocess
import os

class Cam(AbstractCam):
    def __init__(self, video_source):
        super().__init__()
        self._video_source = os.path.join(os.path.dirname(__file__), video_source)
        self._cap = cv2.VideoCapture(self._video_source)
        if not self._cap.isOpened():
            raise ValueError(f"Unable to open video source: {video_source}")

        print("MISB Camera node initialized")
        self._frame = np.zeros((480, 640, 3), np.uint8)
        self._metadata = None
        self._running = False
        self._metacounter = 0

    def start(self):
        self._running = True
        while self._running:
            self.read_frame()
            
            # Code to test frame reading
            cv2.imshow('Display', self._frame)
            if cv2.waitKey(25) == 27:
                break

            # Code to test metadata extraction
            print(self._metadata)


    def stop(self):
        self._running = False
        self._cap.release()

    def read_frame(self):
        ret, frame = self._cap.read()
        if ret:
            self._frame = frame
            self.extract_metadata()
        else:
            self._running = False

    def extract_metadata(self):
        
        # FFMPEG command to extract KLV data
        ffmpeg_command = [
        'ffmpeg',
        '-i', self._video_source,
        '-map', '0:2',
        '-codec', 'copy',
        '-f', 'data',
        '-'
        ]
        process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        # Create a generator for the packets
        if self._metacounter == 0:
            self._packet_generator = klvdata.StreamParser(process.stdout.read())
            self._metacounter = self._metacounter + 1

        # Get the first (top-most) packet
        try:
            top_packet = next(self._packet_generator)
            self._metadata = top_packet.MetadataList()
            
            # Discard the top-most packet by updating the generator
            self._packet_generator = self._packet_generator
        except StopIteration:
            # If there are no packets, set metadata to None
            self._metadata = None
            self._packet_generator = None

    def get_frame(self):
        return self._frame.copy() if self._frame is not None else None

    def get_metadata(self):
        return self._metadata.copy() if self._metadata is not None else None

    def get_xyz(self, x, y):
        if self._metadata and 'SensorLatitude' in self._metadata:
            # This is a simplified example. You'd need to implement
            # the actual conversion based on the specific metadata available
            return (
                self._metadata.get('SensorLatitude', 0),
                self._metadata.get('SensorLongitude', 0),
                self._metadata.get('SensorTrueAltitude', 0)
            )
        return -1, -1, -1

    def get_crop(self, x_lo, y_lo, x_hi, y_hi):
        if self._frame is not None:
            return self._frame[y_lo:y_hi, x_lo:x_hi]
        return None

    # Implement other required methods from AbstractCam...
    def load_color_frame(self, data):
        pass

    def load_depth_frame(self, data):
        pass

    def load_intrinsics(self, camera_info):
        pass

    def transform_pose(self, input_pose, from_frame, to_frame):
        pass

    def get_xyzmap(self, x, y, z):
        pass

    def get_infrared_frame(self):
        pass

    def get_depth(self):
        pass

    def get_intrinsics(self):
        pass

# Usage
if __name__ == "__main__":
    # The video_source could be a file path, RTSP URL, or device number
    video_source = "Truck.ts"
    misb_cam = Cam(video_source)
    
    try:
        misb_cam.start()
    except KeyboardInterrupt:
        # metadata = misb_cam.get_metadata()
        # print(metadata)
        # frames = misb_cam.get_frame()
        # print(frames)
        misb_cam.stop()
    finally:
        cv2.destroyAllWindows()


        # ffmpeg -re -stream_loop -1 -i Truck.ts -map 0 -c copy -c:v h264 -profile baseline -f mpegts 'udp://239.0.0.1:1234?ttl=13'
        # ffmpeg -re -stream_loop -1 -i Truck.ts -map 0 -c copy -f mpegts 'udp://239.0.0.1:1234?ttl=13'