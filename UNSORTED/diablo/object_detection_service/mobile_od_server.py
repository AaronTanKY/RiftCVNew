# from __future__ import print_function
# import time

# import numpy as np

# import grpc
import newMOT_pb2 as pb2
import newMOT_pb2_grpc as pb2_grpc

# import cv2
# from PIL import Image
# import subprocess
# import yaml
# from concurrent import futures
# import logging

# import grpc
# import tensorflow as tf

# import cv2
# import numpy as np

# from pathlib import Path

# import torch

# import base64


# def image_from_bytes(img_byte):
#     jpg_original = base64.b64decode(img_byte)
#     jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
#     frame = cv2.imdecode(jpg_as_np, flags=1)
#     return frame


# import onnx
# import onnxruntime as ort
# import numpy as np
# import tensorflow as tf


# class MobileMOTModel:
#     def __init__(self):
#         MODEL_PATH = "./models/lite-model_ssd_mobilenet_v2_100_int8_default_1.tflite"

#         # # Processed features (copy from Edge Impulse project)
#         # features = [
#         # # <COPY FEATURES HERE!>
#         # ]

#         # Load TFLite model and allocate tensors.
#         self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)

#         # Get input and output tensors.
#         self.input_details = self.interpreter.get_input_details()
#         self.output_details = self.interpreter.get_output_details()
#         self.input_details[0]["shape"] = np.array([1, -1, -1, 3], dtype=np.int32)

#         # Allocate tensors
#         self.interpreter.allocate_tensors()

#         # Print the input and output details of the model
#         print()
#         print("Input details:")
#         print(self.input_details)
#         print()
#         print("Output details:")
#         print(self.output_details)
#         print()

#     def askInfer_MOT(self, request, context):
#         a = request.image  # bytes
#         jpg_original = base64.b64decode(a)
#         jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
#         frame = cv2.imdecode(jpg_as_np, flags=1)

#         # Convert features to NumPy array
#         np_features = np.array(frame)

#         # If the expected input type is int8 (quantized model), rescale data
#         input_type = self.input_details[0]["dtype"]

#         if input_type == np.int8:
#             input_scale, input_zero_point = self.input_details[0]["quantization"]
#             print("Input scale:", input_scale)
#             print("Input zero point:", input_zero_point)
#             print()
#             np_features = (np_features / input_scale) + input_zero_point
#             np_features = np.around(np_features)

#         # Convert features to NumPy array of expected type
#         np_features = np_features.astype(input_type)

#         # Add dimension to input sample (TFLite model expects (# samples, data))
#         np_features = np.expand_dims(np_features, axis=0)

#         # Create input tensor out of raw features
#         self.interpreter.set_tensor(self.input_details[0]["index"], np_features)

#         # Run inference
#         self.interpreter.invoke()

#         # output_details[0]['index'] = the index which provides the input
#         output = self.interpreter.get_tensor(self.output_details[0]["index"])

#         # If the output type is int8 (quantized model), rescale data
#         output_type = self.output_details[0]["dtype"]
#         if output_type == np.int8:
#             output_scale, output_zero_point = self.output_details[0]["quantization"]
#             print("Raw output scores:", output)
#             print("Output scale:", output_scale)
#             print("Output zero point:", output_zero_point)
#             print()
#             output = output_scale * (output.astype(np.float32) - output_zero_point)

#         # Print the results of inference
#         output_dict = {
#             "num_detections": int(self.interpreter.get_tensor(self.outputs[3]["index"])),
#             "detection_classes": self.interpreter.get_tensor(self.outputs[1]["index"]).astype(
#                 np.uint8
#             ),
#             "detection_boxes": self.interpreter.get_tensor(self.outputs[0]["index"]),
#             "detection_scores": self.interpreter.get_tensor(self.outputs[2]["index"]),
#         }
#         print(output_dict)


# def serve(port):
#     server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
#     pb2_grpc.add_MOTorServicer_to_server(MobileMOTModel(), server)
#     server.add_insecure_port("[::]:" + str(port))
#     server.start()
#     print("Server started, listening on " + str(port))
#     server.wait_for_termination()


# def main():
#     with Path("CONFIG.yml").open() as f:
#         CFG = yaml.load(f, Loader=yaml.Loader)
#     cmd = "fuser -k " + str(CFG["MOT_PORT"]) + "/tcp"
#     print(cmd)
#     subprocess.run(cmd, shell=True)
#     time.sleep(0.3)

#     logging.basicConfig()
#     serve(CFG["MOT_PORT"])


# if __name__ == "__main__":
#     main()


def main():
    pass
