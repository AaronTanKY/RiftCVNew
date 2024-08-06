# ******************************************************************************
# Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
#
# This file is part of the example-applications (LRE) product,
# and is released under the Apache 2.0 License.
# *****************************************************************************/

# N: Skipping acquire of configured file 'main/binary-i386/Packages' as repository 'https://apt.latentai.io stable InRelease' doesn't support architecture 'i386'

#!/usr/bin/env python

# Previous imports
import logging
import os
import sys
import grpc
from concurrent import futures
import time
import yaml
from pathlib import Path
import psutil

import torch as T
import cv2 
import numpy as np
import base64

# For watching over new .so files
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import signal

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "protobufs"))
import flask_object_detection_pb2 as pb2
import flask_object_detection_pb2_grpc as pb2_grpc
sys.path.pop()

# Ensure that the import statement matches the filename and class name
from utils import utils
from utils import detector_preprocessor, detector_postprocessor, utils
import json
from pylre import LatentRuntimeEngine

# Get the path names and everything when the script just starts running
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser(description="Run inference")
parser.add_argument(
    "--precision", 
    type=str, 
    default="int8", 
    help="Set precision to run LRE."
)
parser.add_argument(
    "--model_binary_path", 
    type=str, 
    default="../Algorithm-Module/Int8-optimize",      ### TODO: RENAME MODEL TO NAME USED
    help="Path to LRE object directory."
)
parser.add_argument(
    "--input_image_path",
    type=str,
    default="../../sample_images/bus.jpg",              ### TODO: THIS SHOULD THE THE INPUT FRAME
    help="Path to input image.",
)
parser.add_argument(
    "--labels",
    type=str,
    default="../Algorithm-Module/coco.txt",
    help="Path to labels text file.",
)
parser.add_argument(
    "--iterations",
    type=int,
    default=10,
    help="Iterations to average timing.",
)
parser.add_argument(
    "--maximum_detections",
    type=int,
    default=10,
    help="Maximum detections per image.",
)
parser.add_argument(
    "--top_k",
    type=int,
    default=1000,
    help="Top K to be considered.",
)
parser.add_argument(
    "--confidence_threshold",
    type=float,
    default=0.6,
    help="Prediction confidence threshold.",
)
parser.add_argument(
    "--iou_threshold",
    type=float,
    default=0.45,
    help="IOU threshold.",
)
parser.add_argument(
    "--model_family",
    type=str,
    help="Model model_family to use for preprocessing.",
)

args = parser.parse_args()


t_preprocessing = utils.Timer()
t_inference = utils.Timer()
t_postprocessing = utils.Timer()

class ModelFileHandler(FileSystemEventHandler):
    def __init__(self, callback):
        self.callback = callback

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.so'):
            print(f"New .so file detected: {event.src_path}")
            self.callback(event.src_path)

class objDetector(pb2_grpc.objDetectorServicer):
    def __init__(self, model_path):
        # Load model path
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        # Load runtime
        self.lre = LatentRuntimeEngine(str(Path(args.model_binary_path) / "modelLibrary.so"))
        print(self.lre.get_metadata())

        # Set precision
        self.lre.set_model_precision(args.precision)

        # Read metadata from runtime
        self.layout_shapes = utils.get_layout_dims(self.lre.input_layouts, self.lre.input_shapes)
        self.input_size = (self.layout_shapes[0].get('H'), self.layout_shapes[0].get('W'))
        
        self.config = utils.set_processor_configs(args.model_binary_path)
        
        # Load labels
        self.labels = utils.load_labels(args.labels)

    def askInfer_objDetect(self, request, context):
        # Load Image 
        a = request.name  # bytes
        jpg_original = base64.b64decode(a)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        frame = cv2.imdecode(jpg_as_np, flags=1)
        
        
        image = utils.load_image(frame, self.config)
        
        # Finding out image size, but seems useless
        if self.config.visualization_library_cv2:
            image_shape = image.shape
        else:
            image_shape = image.size
    
        # Warm up
        # Is this needed for CPU?
        self.lre.warm_up(10)
        
        iterations = args.iterations

        for i in range(iterations):
            # Pre-process
            t_preprocessing.start()
            if self.config.use_albumentations_library:
                sized_image, transformed_image = detector_preprocessor.preprocess_transforms_albumentations(image, args.model_binary_path)
            else:
                if args.model_family:
                    sized_image, transformed_image = detector_preprocessor.preprocess_transforms(image, args.model_family, self.input_size, self.config)
                else:
                    raise RuntimeError(f"--model_family argument is not provided to preprocess without albumentations.")
            t_preprocessing.stop()

            # Run inference
            t_inference.start()
            self.lre.infer(transformed_image)
            t_inference.stop()

            # Get outputs as a list of PyDLPack
            outputs = self.lre.get_outputs()
            output = outputs[0]

            # Post-process  
            t_postprocessing.start()
            postprocessor_path = Path(args.model_binary_path) / "processors" / "general_detection_postprocessor.py"
            if os.path.exists(postprocessor_path) and self.config.use_albumentations_library:
                postprocessor_path = postprocessor_path.resolve()
                postprocessor_path = str(postprocessor_path.parents[0])
                sys.path.append(postprocessor_path)
                from general_detection_postprocessor import post_process
                import torch as T
                output_torch = T.from_dlpack(output)
                output = post_process(output_torch, max_det_per_image=args.maximum_detections, prediction_confidence_threshold=args.confidence_threshold, iou_threshold=args.iou_threshold, k=args.top_k)
            else:
                output = detector_postprocessor.postprocess(output, max_det_per_image=args.maximum_detections, prediction_confidence_threshold=args.confidence_threshold, iou_threshold=args.iou_threshold, k=args.top_k, config=self.config)
            t_postprocessing.stop()

            # Visualize
            output_image = utils.plot_boxes(sized_image, output, self.labels)

            _, buffer = cv2.imencode('.jpg', output_image)
            data = base64.b64encode(buffer)

            '''
            ### USELESS CODE FOR NOW

            ## OUTPUT FILE
            output_filename = utils.save_image(output_image, args.input_image_path, config)
    
            # Get the average elapsed time in milliseconds
            average_preprocessing_time = t_preprocessing.averageElapsedMilliseconds()
            std_dev_preprocessing = t_preprocessing.standardDeviationMilliseconds()
            
            average_inference_time = t_inference.averageElapsedMilliseconds()
            std_dev_inference = t_inference.standardDeviationMilliseconds()
            
            average_postprocessing_time = t_postprocessing.averageElapsedMilliseconds()
            std_dev_postprocessing = t_postprocessing.standardDeviationMilliseconds()

            average_time = average_preprocessing_time + average_inference_time + average_postprocessing_time

            # Create a dictionary representing the model details
            j = {
                "UUID": self.lre.model_id,
                "Precision": self.lre.model_precision,
                "Device": self.lre.device_type,
                "Input Image Size": image_shape,
                "Model Input Shapes": self.lre.input_shapes,
                "Model Input Layouts": self.lre.input_layouts,
                "Average Preprocessing Time ms": {
                    "Mean": utils.roundToDecimalPlaces(average_preprocessing_time, 3),
                    "std_dev": utils.roundToDecimalPlaces(std_dev_preprocessing, 3)
                },
                "Average Inference Time ms": {
                    "Mean": utils.roundToDecimalPlaces(average_inference_time, 3),
                    "std_dev": utils.roundToDecimalPlaces(std_dev_inference, 3)
                },
                "Average Total Postprocessing Time ms": {
                    "Mean": utils.roundToDecimalPlaces(average_postprocessing_time, 3),
                    "std_dev": utils.roundToDecimalPlaces(std_dev_postprocessing, 3)
                },
                "Total Time ms": utils.roundToDecimalPlaces(average_time, 3),
                "Annotated Image": output_filename
            }

            json_str = json.dumps(j, indent=2)
            print(json_str)
            '''

            return pb2.objDetect_reply(message=data, x_lo=str(_x_low=0), y_lo=str(_y_low=0), x_hi=str(_x_high=0), y_hi=str(_y_high=0))

class Server:
    def __init__(self):
        self.server = None
        self.is_running = False
        self.model_path = self.get_latest_model()

    def get_latest_model(self):
        model_dir = os.path.dirname(args.model_binary_path)
        so_files = [f for f in os.listdir(model_dir) if f.startswith('modelLibrary') and f.endswith('.so')]
        if not so_files:
            return os.path.join(args.model_binary_path, "modelLibrary.so")
        latest_model = max(so_files, key=lambda x: int(x.replace('modelLibrary', '').replace('.so', '') or 0))
        return os.path.join(model_dir, latest_model)

    def start(self):
        self.is_running = True
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        pb2_grpc.add_objDetectorServicer_to_server(objDetector(self.model_path), self.server)
        port = self.get_available_port()
        self.server.add_insecure_port(f'[::]:{port}')
        self.server.start()
        print(f"Server started, listening on port {port}")

    def stop(self):
        if self.server:
            self.server.stop(0)
        self.is_running = False
        print("Server stopped")

    def restart(self):
        self.stop()
        time.sleep(1)  # Wait for the server to fully stop
        self.model_path = self.get_latest_model()
        self.start()

    def get_available_port(self):
        yaml_source = os.path.join(os.path.dirname(__file__), "CONFIG.yml")
        with Path(yaml_source).open() as f:
            CFG = yaml.load(f, Loader=yaml.Loader)
        port = int(CFG["OBJDETECT_PORT"])
        kill_process_on_port(port)
        return port

def model_update_callback(new_model_path):
    global server
    print(f"Restarting server with new model: {new_model_path}")
    server.restart()

def signal_handler(signum, frame):
    global server
    print("Stopping server...")
    server.stop()
    sys.exit(0)

if __name__ == '__main__':
    logging.basicConfig()
    
    server = Server()
    server.start()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    event_handler = ModelFileHandler(model_update_callback)
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(args.model_binary_path), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()