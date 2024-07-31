from concurrent import futures
import logging

import grpc

import psutil

import base64
import cv2
import logging
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the current script
sys.path.append(os.path.join(dir_path, "protobufs"))
import objDetect_pb2 as pb2
import objDetect_pb2_grpc as pb2_grpc

import cv2

import base64
import numpy as np

from pathlib import Path

import cv2
import numpy as np

import torch

import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the current script
head, tail = os.path.split(dir_path)
dir_path = os.path.join(head, 'Algorithm-Module\yolov7-main')
sys.path.append(os.path.join(dir_path, 'utils'))

# OBJ DETECTION
from general import check_requirements, set_logging
from google_utils import attempt_download
from torch_utils import select_device
sys.path.pop()

sys.path.append(os.path.join(dir_path, 'models'))
from yolo import Model
sys.path.pop()

import yaml
import subprocess

import time
from pathlib import Path, PurePath

from dataclasses import dataclass 

@dataclass
class SingleObjectDetectionResult():
    class_: int
    x_low: float
    x_high: float
    y_low: float
    y_high: float

# type name = Callable[]

class objDetector(pb2_grpc.objDetectorServicer):

    def __init__(self):  #def make_objDetect(path_or_model='path/to/model.pt', autoshape=True):  #model str dict nn module shd work
        autoshape = 1
        self_path = Path(__file__).parent.parent.resolve()
        path_of_model = str(self_path) + "\Algorithm-Module\yolov7.pt" # r'/home/nuc/catkin_ws/src/realsense_cv/yolov7.pt'#C:\Users\User\Desktop\cvbotbot\yolov7.pt'
        model = Model()
        model.load_state_dict(torch.load(path_of_model))
        # model = torch.load(path_of_model, map_location=torch.device('cpu')) if isinstance(path_of_model, str) else path_of_model  # load checkpoint
        if isinstance(model, dict):
            model = model['ema' if model.get('ema') else 'model']  # load model

        hub_model = Model(model.yaml).to(next(model.parameters()).device)  # create
        hub_model.load_state_dict(model.float().state_dict())  # load state_dict
        hub_model.names = model.names  # class names
        if autoshape:
            hub_model = hub_model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
        device = select_device('0' if torch.cuda.is_available() else 'cpu')  # default to GPU if available

        self.model_objDetect = hub_model.to(device)


    def askInfer_objDetect(self, request, context): #request.name set in client as you 
        a = request.name #bytes
        with torch.no_grad():    
            jpg_original = base64.b64decode(a)
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            frame = cv2.imdecode(jpg_as_np, flags=1)
            
            results = self.model_objDetect(frame) #infer
            results.print()

            pred = results.pandas().xyxy[0]
            print(pred)

            centered = False #for drawing and tracking only 1 guy
            x_mid, y_mid = 0,0
            _x_low, _y_low, _x_high, _y_high = 0,0,0,0

            for index, row in pred.iterrows(): #!For each Person detected in 1 frame
                if int(row['class']) == 0 and float(row['confidence']) > 0.5: 
                    frame = cv2.rectangle(frame, (int(row['xmin']), int(row['ymin'])),(int(row['xmax']),int(row['ymax'])), (255, 0, 0), 10) #visualiser

                    if not centered: #first one for now(ie. likely the oldest)
                        _x_low = int(row['xmin'])
                        _y_low = int(row['ymin'])
                        _x_high = int(row['xmax'])
                        _y_high = int(row['ymax'])

                        x_mid = (_x_low+_x_high)/2
                        y_mid = (_y_low+_y_high)/2

                        frame = cv2.circle(frame, (int(x_mid),int(y_mid)), radius=10, color=(0, 0, 255), thickness=-1)
                        centered = True

                        # stuff_here = machine_learning


                # if float(row['confidence']) > 0.7: 
            
            _, buffer = cv2.imencode('.jpg', frame)
            data = base64.b64encode(buffer)
            return pb2.objDetect_reply(message= data, x_lo = str(_x_low), y_lo = str(_y_low), x_hi = str(_x_high), y_hi = str(_y_high))
    
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
    port = int(CFG["CAMERA_PORT"])
    kill_process_on_port(port)

    time.sleep(0.1)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_objDetectorServicer_to_server(objDetector(), server)
    server.add_insecure_port('[::]:' + str(CFG['OBJDETECT_PORT']))
    server.start()
    print("Server started, listening on " + str(CFG['OBJDETECT_PORT']))
    server.wait_for_termination()

    # start the process

if __name__ == '__main__':
    logging.basicConfig()
    serve()