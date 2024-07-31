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
import os
import sys
from numpy import random
import time
import yaml
import subprocess
from dataclasses import dataclass
from latentai import LEIPInference
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import scale_coords, xyxy2xywh
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

# Initialize StrongSORT
self_path = Path(__file__).parent.resolve()
config_strongsort = str(self_path / "strong_sort/configs/strong_sort.yaml")
cfg = get_config()
cfg.merge_from_file(config_strongsort)
strong_sort_weights = str(self_path / "osnet_x0_25_market1501.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
half = torch.cuda.is_available()
imgsz = (640, 640)

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
strongsort_list[0].model.warmup()
outputs, prev_frames = [None] * 1, [None] * 1

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
        self.model = LEIPInference.load_model("yolov7_optimized.leip")

    def askInfer_MOT(self, request, context):
        a = request.image  # bytes
        jpg_original = base64.b64decode(a)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        frame = cv2.imdecode(jpg_as_np, flags=1)

        # Adjust for inference
        img = letterbox(frame, imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        im = np.ascontiguousarray(img)
        im = im.astype('float32') / 255.0
        im = im[np.newaxis, ...]

        # Inference
        pred = self.model.run(im)
        pred = non_max_suppression(pred, 0.5, 0.45, 0, False)  # conf_thres, iou_thres, 0, False

        list_of_detections = []

        for i, det in enumerate(pred):  # detections per image
            if cfg.STRONGSORT.ECC:
                strongsort_list[i].tracker.camera_update(prev_frames[i], frame)
            if det is not None and len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], frame.shape).round()
                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]
                try:
                    outputs[i] = strongsort_list[i].update(
                        xywhs, confs, clss, frame
                    )
                except Exception as e:
                    print(e)

                if len(outputs) > 0:
                    for _, (output, confidence) in enumerate(zip(outputs[i], confs)):
                        class_ = int(output[5])
                        id = int(output[4])
                        x_low, y_low, x_high, y_high = output[:4]

                        single_detection = {
                            "x_low": x_low,
                            "y_low": y_low,
                            "x_high": x_high,
                            "y_high": y_high,
                            "id": id,
                            "class": class_,
                            "confidence": confidence,
                        }
                        list_of_detections.append(single_detection)

            else:
                strongsort_list[i].increment_ages()
                print("no_detections")
            prev_frames[i] = frame

        _, buffer = cv2.imencode(".jpg", frame)
        data = base64.b64encode(buffer)

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
    logging.basicConfig()
    serve(CFG["MOT_PORT"])