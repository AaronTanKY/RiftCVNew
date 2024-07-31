import cv2
import numpy as np
from MPCam import AbstractCam
import klvdata
from klvdata import misb0601 
import subprocess
import os
import threading

class Cam(AbstractCam):
    def __init__(self, video_source):
        super().__init__()
        self._video_source = video_source
        self._cap = None
        self._frame = np.zeros((480, 640, 3), np.uint8)
        self._metadata = None
        self._running = False
        self._metacounter = 0
        self._thread = None

    def start(self):
        
        if not self._thread:
            self._running = True
            self._thread = threading.Thread(target=self._run)
            self._thread.daemon = True
            self._thread.start()
        
        '''
        self._running = True
        while self._running:
            self.read_frame()
            
            
            # Code to test frame reading
            cv2.imshow('Display', self._frame)
            if cv2.waitKey(25) == 27:
                break

            # Code to test metadata extraction
            print(self._metadata)
            
        '''
    
    def _run(self):
        while self._running:
            print("Initializing MISB Node")
            self._cap = cv2.VideoCapture(self._video_source)

            import time
            time.sleep(2)

            while not self._cap.isOpened():
                print("Failed to open video source, retrying...")
                self._cap = cv2.VideoCapture(self._video_source)
            print("MISB Node Initialized")

            while self._running:
                self.read_frame()
                if self._frame is not None:
                    cv2.imshow('Display', self._frame)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
            '''
            self.read_frame()
            cv2.imshow('Display', self._frame)
            if cv2.waitKey(25) == 27:
                break
            '''


    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
        self._cap.release()

    def read_frame(self):
        ret, frame = self._cap.read()
        if ret:
            self._frame = frame
            self.extract_metadata()
            print(self._metadata)
        else:
            self._running = False

    def extract_metadata(self):
        # FFMPEG command to extract KLV data from a UDP stream
        ffmpeg_command = [
            'ffmpeg',
            '-i', f"{self._video_source}?fifo_size=50000000&overrun_nonfatal=1",
            '-map', '0:2',
            '-codec', 'copy',
            '-f', 'data',
            '-'
        ]
        process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        # Create a generator for the packets
        if self._metacounter == 0:
            self._packet_generator = klvdata.StreamParser(process.stdout)
            self._metacounter = self._metacounter + 1

        # Get the first (top-most) packet
        try:
            top_packet = next(self._packet_generator)
            self._metadata = top_packet.MetadataList()
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
    # The video_source is now a UDP URL
    video_source = "udp://239.0.0.1:1234"
    misb_cam = Cam(video_source)
    
    try:
        misb_cam.start()
    except KeyboardInterrupt:
        misb_cam.stop()
    finally:
        cv2.destroyAllWindows()
    