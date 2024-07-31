import cv2
import numpy as np
from MPCam import AbstractCam
import klvdata
from klvdata import misb0601  # For parsing KLV metadata
import subprocess
import os

class MISBCam(AbstractCam):
    def __init__(self, video_source):
        super().__init__()
        self.video_source = video_source  # Add this line
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video source: {video_source}")

        print("MISB Camera node initialized")        
        self._frame = np.zeros((480, 640, 3), np.uint8)
        self._metadata = None
        self._running = False

    def start(self):
        self._running = True
        while self._running:
            self._read_frame()
            
            # Code to test frame reading
            cv2.imshow('Display', self._frame)
            if cv2.waitKey(25) == 27:
                break

            # Code to test metadata extraction
            print(self._metadata)

    def stop(self):
        self._running = False
        self.cap.release()

    def _read_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self._frame = frame
            self._extract_metadata()
        else:
            self._running = False

    def _extract_metadata(self):
        
        # FFMPEG command to extract KLV data
        ffmpeg_command = [
        'ffmpeg',
        '-i', self.video_source,
        '-map', '0:2',
        '-codec', 'copy',
        '-f', 'data',
        '-'
        ]
        process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        # Store metadata in packets into self._metadata
        for packet in klvdata.StreamParser(process.stdout.read()):
            self._metadata = packet.MetadataList()
            # metadata = packet.MetadataList()
            # for key, value in metadata.items():
                # print(key, value)
            
            # packet.structure()
            # count = count + 1
            # print(count)
    '''
    def _parse_klv(self, klv_data):
        metadata = {}
        try:
            for packet in misb0601.UASLocalMetadataSet(klv_data):
                metadata[packet.key] = packet.value
        except Exception as e:
            print(f"Error parsing MISB 0601 data: {e}")
        return metadata
    '''

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
    video_source = os.path.join(os.path.dirname(__file__), "Truck.ts")
    video_source = 'Truck.ts'
    misb_cam = MISBCam(video_source)
    
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