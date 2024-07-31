from __future__ import print_function

from concurrent import futures
import logging

import grpc
import newMOT_pb2 as pb2
import newMOT_pb2_grpc as pb2_grpc

import cv2
import base64
import numpy as np

from pathlib import Path

import torch
from numpy import random

import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the current script
sys.path.append(os.path.join(dir_path, "yolov7Original"))
from utils.general import set_logging
from utils.datasets import letterbox

sys.path.pop()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

import os
from pathlib import Path, PurePath

self_path = PurePath(__file__).parents[0]

p = str(self_path) + str("/Yolov7_StrongSORT_OSNet")
os.chdir("Yolov7_StrongSORT_OSNet")  # r)
# r"/home/nuc/catkin_ws/src/realsense_cv/Yolov7_StrongSORT_OSNet")
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch.backends.cudnn as cudnn


from pathlib import Path, PurePath

self_path = PurePath(__file__).parents[0]
p = str(self_path) + str("/Yolov7_StrongSORT_OSNet")
ROOT = Path(
    os.path.relpath(p), Path.cwd()
)  # r"/home/nuc/catkin_ws/src/realsense_cv/Yolov7_StrongSORT_OSNet")
WEIGHTS = ROOT / "weights"

# yes, this is not used in the code
# no, it is not removable (it DIE)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / "yolov7") not in sys.path:
    sys.path.append(str(ROOT / "yolov7"))  # add yolov5 ROOT to PATH
if str(ROOT / "strong_sort") not in sys.path:
    sys.path.append(str(ROOT / "strong_sort"))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from yolov7.models.experimental import attempt_load

# from yolov7.utils.datasets import LoadImages, LoadStreams, letterbox #using other letterbox instead
from yolov7.utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
    check_requirements,
    cv2,
    check_imshow,
    xyxy2xywh,
    increment_path,
    strip_optimizer,
    colorstr,
    check_file,
)
from yolov7.utils.torch_utils import select_device, time_synchronized
from yolov7.utils.plots import plot_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

cudnn.benchmark = True

##### initialize StrongSORT ######
self_path = Path(__file__).parent.resolve()  # PurePath(__file__).parents[0]
p = str(self_path) + str("/strong_sort/configs/strong_sort.yaml")
config_strongsort = p  # Path(r"/home/nuc/catkin_ws/src/realsense_cv/Yolov7_StrongSORT_OSNet/strong_sort/configs/strong_sort.yaml")
cfg = get_config()
cfg.merge_from_file(config_strongsort)  # opt.

p = str(self_path) + str("/osnet_x0_25_market1501.pt")
strong_sort_weights = p  # Path(r"/home/nuc/catkin_ws/src/realsense_cv/Yolov7_StrongSORT_OSNet/osnet_x0_25_market1501.pt")#WEIGHTS / 'osnet_x0_25_market1501.pt', #'osnet_x0_25_msmt17.pt',  # model.pt path,

if torch.cuda.is_available():
    half = True  # half precision only supported on CUDA
else:
    half = False

imgsz = (640, 640)  # inference size (height, width)

os.chdir(self_path.parent)
set_logging()
# Dataloader
import time

import yaml
import subprocess

from dataclasses import dataclass


@dataclass
class SingleMOTDetectionResults:
    frame_no: int
    id: int
    x_low: float
    x_high: float
    y_low: float
    y_high: float
    class_: int
    confidence: int


class MOTor(pb2_grpc.MOTorServicer):
    def __init__(self):
        self._path = Path(__file__).parent.resolve()
        p = (
            str(self._path) + "/yolov7-tiny.pt"
        )  # Change to this if you want to run yolov7 instead of yolov7-tiny"/yolov7.pt"
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_MOT = attempt_load(p, map_location=self._device)

        WEIGHTS.mkdir(parents=True, exist_ok=True)  # load FP32 model
        (self._names,) = (self.model_MOT.names,)
        self._colors = [[random.randint(0, 255) for _ in range(3)] for _ in self._names]
        stride = int(
            self.model_MOT.stride.max()
        )  # int added by YL  # model stride (https://github.com/mikel-brostrom/Yolov7_StrongSORT_OSNet/issues/2)
        self._imgsz = check_img_size(imgsz[0], s=stride)

    def askInfer_MOT(self, request, context):  # request.name set in client as you
        with torch.no_grad():
            a = request.image  # bytes

            # frame = self.imgFromBytes(a) #for some reason putting it in func caused it to complain abt 2 arguments even when there were only 1
            jpg_original = base64.b64decode(a)
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            frame = cv2.imdecode(jpg_as_np, flags=1)

            # adjust for inference
            img = letterbox(frame, imgsz)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            im = np.ascontiguousarray(img)
            im = torch.from_numpy(im).to(self._device)
            im = im.float()  # uint8 to fp16/32
            im /= 255.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            # Inference
            pred = self.model_MOT(im)
            pred = non_max_suppression(
                pred[0], 0.5, 0.45, 0, False
            )  # conf_thres, iou_thres, 0, False

            _x_low, _y_low, _x_high, _y_high = 0, 0, 0, 0

            list_of_detections = []

            for i, det in enumerate(pred):  # detections per image, DRAW
                if cfg.STRONGSORT.ECC:  # camera motion compensation
                    strongsort_list[i].tracker.camera_update(prev_frames[i], frame)
                if det is not None and len(det):  # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], frame.shape).round()
                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]
                    try:  # rarely, but occassionally,throws out of range, no clue why (TODO)
                        outputs[i] = strongsort_list[i].update(
                            xywhs.cpu(), confs.cpu(), clss.cpu(), frame
                        )  # pass detections to strongsort
                    except:
                        pass
                    print(outputs[0])

                    if len(outputs) > 0:
                        for _, (output, confidence) in enumerate(zip(outputs[i], confs)):
                            class_ = int(output[5])  # integer class
                            id = int(output[4])  # integer id

                            _x_low = output[0]
                            _y_low = output[1]
                            _x_high = output[2]
                            _y_high = output[3]

                            single_detection = {
                                "x_low": _x_low,
                                "y_low": _y_low,
                                "x_high": _x_high,
                                "y_high": _y_high,
                                "id": id,
                                "class": class_,
                                "confidence": confidence,
                            }

                            list_of_detections.append(single_detection)

                else:  # no detections
                    strongsort_list[i].increment_ages()
                    print("no_detections")
                prev_frames[i] = frame

            _, buffer = cv2.imencode(".jpg", frame)
            data = base64.b64encode(buffer)
            # print(x_mid)
            # print(y_mid)

            return pb2.MOT_reply(image=data, detections=list_of_detections)


def serve(port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_MOTorServicer_to_server(MOTor(), server)
    server.add_insecure_port("[::]:" + str(port))
    server.start()
    print("Server started, listening on " + str(port))
    server.wait_for_termination()


if __name__ == "__main__":
    with Path("CONFIG.yml").open() as f:
        CFG = yaml.load(f, Loader=yaml.Loader)
    cmd = "fuser -k " + str(CFG["MOT_PORT"]) + "/tcp"
    print(cmd)
    subprocess.run(cmd, shell=True)
    time.sleep(0.3)

    # Create as many strong sort instances as there are video sources
    strongsort_list = []
    strongsort_list.append(
        StrongSORT(
            strong_sort_weights,
            device,
            half,
            max_dist=cfg.STRONGSORT.MAX_DIST,
            max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
            max_age=cfg.STRONGSORT.MAX_AGE,
            n_init=cfg.STRONGSORT.N_INIT,
            nn_budget=cfg.STRONGSORT.NN_BUDGET,
            mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
            ema_alpha=cfg.STRONGSORT.EMA_ALPHA,
        )
    )
    strongsort_list[0].model.warmup()  # i
    outputs, prev_frames = [None] * 1, [None] * 1

    logging.basicConfig()
    serve(CFG["MOT_PORT"])
