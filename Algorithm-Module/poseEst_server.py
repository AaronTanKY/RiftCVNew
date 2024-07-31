from concurrent import futures
import logging

import grpc
import base64
import cv2

# from __future__ import print_function

import logging

import grpc
import poseEst_pb2 as pb2
import poseEst_pb2_grpc as pb2_grpc

import cv2

import base64
import numpy as np
#client runs in ur browsert_function

import logging
import base64
import numpy as np

#client runs in ur browser
import cv2
import numpy as np

import torch
from torchvision import transforms

import sys

import os
dir_path = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the current script
sys.path.append(os.path.join(dir_path, 'yolov7Original'))
from models.yolo import Model #TODO: Fix this: While not used, removing throws error (model not found)
#pose estimation
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
sys.path.pop()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

from pathlib import Path
import subprocess
import yaml
import time
from pathlib import Path, PurePath

class poseEstor(pb2_grpc.poseEstorServicer):

    def __init__(self): 
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self_path = Path(__file__).parent.resolve()
        p = str(self_path) + "/yolov7-w6-pose.pt"

        model = torch.load(p, map_location=self._device)['model'] #r'/home/nuc/catkin_ws/src/realsense_cv/yolov7-w6-pose.pt'
        model.float().eval()
        if torch.cuda.is_available():
            model.to(self._device)
            model.half() #to float 16
        self.model_poseEst = model
    
    def askInfer_poseEst(self, request, context): #request.name set in client as you 

        a = request.name #img bytes
        jpg_original = base64.b64decode(a)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        frame = cv2.imdecode(jpg_as_np, flags=1)
        
        #infer
        image = letterbox(frame, 960, stride=64, auto=True)[0] # shape: (768, 960, 3)
        image = transforms.ToTensor()(image) # torch.Size([3, 768, 960])
        image = image.unsqueeze(0) # torch.Size([1, 3, 768, 960])=

        if torch.cuda.is_available():
            image = image.half().to(self._device)

        with torch.no_grad():
            output, _ = self.model_poseEst(image) # torch.Size([1, 45900, 57])
        #return output, image
        #results.print() # 2 car, 7 truck, 9 traffic light, 0 person (class)

        output = non_max_suppression_kpt(output, 
                                     0.25, # Confidence Threshold
                                     0.65, # IoU Threshold
                                     nc=self.model_poseEst.yaml['nc'], # Number of Classes
                                     nkpt=self.model_poseEst.yaml['nkpt'], # Number of Keypoints
                                     kpt_label=True)
        with torch.no_grad():
            output = output_to_keypoint(output)
        nimg = image[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2RGBA)
        for idx in range(output.shape[0]):
            plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)


        _, buffer = cv2.imencode('.jpg', nimg)
        data = base64.b64encode(buffer)

        return pb2.poseEst_reply(message= data)
    
       
def serve():
    with Path('CONFIG.yml').open() as f:
        CFG = yaml.load(f,Loader= yaml.Loader)
    cmd = "fuser -k " + str(CFG['POSEEST_PORT']) + "/tcp"
    print(cmd)
    subprocess.run(cmd, shell=True)

    time.sleep(0.1)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_poseEstorServicer_to_server(poseEstor(), server)
    server.add_insecure_port('[::]:' + str(CFG['POSEEST_PORT']))
    server.start()
    print("Server started, listening on " + str(CFG['POSEEST_PORT']))
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    serve()