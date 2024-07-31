import os
import sys
import grpc
import cv2
import base64
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the current script
sys.path.append(os.path.join(dir_path, "protobufs"))
import main_server2_pb2 as pb2
import main_server2_pb2_grpc as pb2_grpc

def run():
    channel = grpc.insecure_channel('localhost:50050')
    stub = pb2_grpc.GreeterStub(channel)

    while True:
        # Test get_misb_metadata
        print("Testing get_misb_metadata...")
        metadata_response = stub.get_metadata(pb2.HelloRequest())

        frame_response = stub.get_rframe(pb2.HelloRequest(name="y"))
        frame = imgFromBytes(frame_response.message)
        cv2.imshow('FRAME', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 

        # Print the metadata
        for key, item in metadata_response.items.items():
            print(f"{key}:")
            print(f"  Name: {item.name}")
            print(f"  Description: {item.description}")
            print(f"  Alternate Name: {item.alternate_name}")
            if item.HasField('string_value'):
                print(f"  Value: {item.string_value}")
            elif item.HasField('double_value'):
                print(f"  Value: {item.double_value}")
            elif item.HasField('timestamp_value'):
                print(f"  Value: {item.timestamp_value.ToDatetime()}")
            print()

def imgFromBytes(img_byte):
    jpg_original = base64.b64decode(img_byte)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    frame = cv2.imdecode(jpg_as_np, flags=1)
    return frame

if __name__ == '__main__':
    run()