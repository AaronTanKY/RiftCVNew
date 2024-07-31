import cv2
import numpy as np
from MPCam import AbstractCam
import klvdata
import subprocess
import threading
import queue
import pickle

class Cam(AbstractCam):
    def __init__(self, video_source):
        super().__init__()
        self._video_source = video_source
        self._cap = None
        self._frame = np.zeros((480, 640, 3), np.uint8)
        self._metadata = None
        self._running = False
        self._frame_thread = None
        self._metadata_thread = None
        self._frame_queue = queue.Queue(maxsize=1)
        self._metadata_queue = queue.Queue(maxsize=1)

    def start(self):
        if not self._frame_thread:
            self._running = True
            self._frame_thread = threading.Thread(target=self._run)
            self._frame_thread.daemon = True
            self._frame_thread.start()
            self._metadata_thread = threading.Thread(target=self._process_metadata)
            self._metadata_thread.daemon = True
            self._metadata_thread.start()

    def _run(self):
        print("Initializing MISB Node")
        self._cap = cv2.VideoCapture(self._video_source)
        while not self._cap.isOpened():
            print("Failed to open video source, retrying...")
            self._cap = cv2.VideoCapture(self._video_source)
        print("Video source opened")

        while self._running:
            ret, frame = self._cap.read()
            if ret:
                self._frame = frame
            else:
                self._running = False
            #self.display_frame()

    def stop(self):
        self._running = False
        if self._frame_thread:
            self._frame_thread.join()
        if self._metadata_thread:
            self._metadata_thread.join()
        if self._cap:
            self._cap.release()

    def display_frame(self):
        cv2.imshow('Display', self._frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            self.stop()

    def extract_metadata(self):
        ffmpeg_command = [
            'ffmpeg',
            '-i', f"{self._video_source}?fifo_size=50000000&overrun_nonfatal=1",
            '-map', '0:2',
            '-codec', 'copy',
            '-f', 'data',
            '-'
        ]
        process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return process.stdout

    def _process_metadata(self):
        print("Retrieving Metadata...")
        metadata_stream = self.extract_metadata()
        while self._running:
            try:
                packet = metadata_stream.read(188)
                if not packet:
                    # End of stream, restart
                    metadata_stream = self.extract_metadata()
                    continue
                self._metadata = packet
            except Exception as e:
                print(f"Error processing metadata: {e}")
                metadata_stream = self.extract_metadata()

    ### GET FUNCTIONS
    def get_frame(self):
        if self._frame is not None:
            return self._frame 
        return None

    def get_metadata(self):
        if self._metadata is not None:
            return self._metadata
        return None

    def get_xyz(self, x, y):
        if self._metadata and 'SensorLatitude' in self._metadata:
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
    video_source = "udp://239.0.0.1:1234"
    misb_cam = Cam(video_source)
    
    try:
        misb_cam.start()
        import time
        while True:
            #cv2.imshow('Display', misb_cam.get_frame())
            print(misb_cam.get_metadata())
    except KeyboardInterrupt:
        misb_cam.stop()
    finally:
        cv2.destroyAllWindows()
        
        
# ffmpeg -re -stream_loop -1 -i Truck.ts -map 0 -c copy -f mpegts 'udp://239.0.0.1:1234?ttl=13'