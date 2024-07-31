import logging
import os
import sys
import grpc
from concurrent import futures
import time
import yaml
from pathlib import Path
import psutil

import torch
import cv2
import numpy as np
import base64

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "protobufs"))
import flask_object_detection_pb2 as pb2
import flask_object_detection_pb2_grpc as pb2_grpc

class objDetector(pb2_grpc.objDetectorServicer):
    def __init__(self):
        if torch.cuda.is_available():
            print("Using CUDA")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.to(self.device)
        self.model.eval()

    def askInfer_objDetect(self, request, context):
        a = request.name  # bytes
        jpg_original = base64.b64decode(a)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        frame = cv2.imdecode(jpg_as_np, flags=1)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform inference
        results = self.model(frame_rgb)
        
        detections = results.xyxy[0].cpu().numpy()
        
        _x_low, _y_low, _x_high, _y_high = 0, 0, 0, 0
        for detection in detections:
            x_low, y_low, x_high, y_high, conf, class_id = detection
            if conf > 0.5:  # Confidence threshold
                cv2.rectangle(frame, (int(x_low), int(y_low)), (int(x_high), int(y_high)), (255, 0, 0), 2)
                cv2.putText(frame, f'{results.names[int(class_id)]}: {conf:.2f}', (int(x_low), int(y_low) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                
                if _x_low == 0:
                    _x_low, _y_low, _x_high, _y_high = int(x_low), int(y_low), int(x_high), int(y_high)

        _, buffer = cv2.imencode('.jpg', frame)
        data = base64.b64encode(buffer)
        return pb2.objDetect_reply(message=data, x_lo=str(_x_low), y_lo=str(_y_low), x_hi=str(_x_high), y_hi=str(_y_high))


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
       
def serve(): 
    yaml_source = os.path.join(os.path.dirname(__file__), "CONFIG.yml")
    with Path(yaml_source).open() as f:
        CFG = yaml.load(f,Loader= yaml.Loader)
        
    ## FOR LINUX or MACOS
    '''
    cmd = "fuser -k " + str(CFG['OBJDETECT_PORT']) + "/tcp"
    print(cmd)
    subprocess.run(cmd, shell=True)
    '''

    ## FOR WINDOWS
    port = int(CFG["OBJDETECT_PORT"])
    kill_process_on_port(port)
    time.sleep(0.5)
    port = str(port)

    time.sleep(0.1)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_objDetectorServicer_to_server(objDetector(), server)
    server.add_insecure_port('[::]:' + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()

    # start the process

if __name__ == '__main__':
    logging.basicConfig()
    serve()