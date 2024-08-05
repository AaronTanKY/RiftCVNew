## 22/7
from __future__ import print_function
import traceback
import logging

import grpc

import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the current script
sys.path.append(os.path.join(dir_path, "protobufs"))

import main_server2_pb2 as pb2
import main_server2_pb2_grpc as pb2_grpc

import flask_object_detection_pb2 as od_pb2
import flask_object_detection_pb2_grpc as od_pb2_grpc

sys.path.pop()

import cv2

import base64
import numpy as np

from flask import Flask, Response

import time

import paho.mqtt.client as mqtt

import subprocess
import yaml
from pathlib import Path

import json
import math

import bisect
from collections import deque

import psutil 

import ffmpeg
import subprocess
import threading
import queue

import klvdata.common as common
from datetime import datetime

prevtime = time.time()
t = time.time()
sent_intrin = False

def timestamp(msg):
    global prevtime, t

    prevtime = t
    timediff = time.time() - prevtime

    current_time = str(timediff)  # time.strftime("%H:%M:%S.%f", t)
    if msg != "x":
        print(msg + current_time)
    t = time.time()


# def is_similar(image1, image2):
#     return image1.shape == image2.shape and not(np.bitwise_xor(image1,image2).any())
# 10.255.252.61

# source /opt/ros/<distro>/setup.bash
# roslaunch realsense2_camera rs_camera.launch align_depth:=true


# watch -n 0.01 date +"%s.%N"
# fuser -k 8080/tcp

"""
timing of getting frame ofn separate thread
then between publishing and receiving
publishing shd be timechip with end of xyz frame to displayed time
"""

app = Flask(__name__)

@app.route("/test")
def test_route():
    return "Test working"

@app.route("/")
def index():
    return "/feed for the image; /objDetect for the image with objDetect overlays"


def imgFromBytes(img_byte):
    try:
        jpg_original = base64.b64decode(img_byte)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        frame = cv2.imdecode(jpg_as_np, flags=1)
        if frame is None:
            raise ValueError("Failed to decode image")
        return frame
    except Exception as e:
        print(f"Error in imgFromBytes: {e}")
        return None


def imgToBytes(img):
    _, buffer = cv2.imencode(".jpg", img)
    data = base64.b64encode(buffer)
    return data


# Buffer to store recent metadata
metadata_buffer = deque(maxlen=100)  # Adjust size as neededfrom __future__ import print_function
import traceback
import logging

import grpc

import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the current script
sys.path.append(os.path.join(dir_path, "protobufs"))

import main_server2_pb2 as pb2
import main_server2_pb2_grpc as pb2_grpc

import flask_object_detection_pb2 as od_pb2
import flask_object_detection_pb2_grpc as od_pb2_grpc

sys.path.pop()

import cv2

import base64
import numpy as np

from flask import Flask, Response

import time

import paho.mqtt.client as mqtt

import subprocess
import yaml
from pathlib import Path

import json
import math

import bisect
from collections import deque

import psutil 

prevtime = time.time()
t = time.time()
sent_intrin = False

def timestamp(msg):
    global prevtime, t

    prevtime = t
    timediff = time.time() - prevtime

    current_time = str(timediff)  # time.strftime("%H:%M:%S.%f", t)
    if msg != "x":
        print(msg + current_time)
    t = time.time()


# def is_similar(image1, image2):
#     return image1.shape == image2.shape and not(np.bitwise_xor(image1,image2).any())
# 10.255.252.61

# source /opt/ros/<distro>/setup.bash
# roslaunch realsense2_camera rs_camera.launch align_depth:=true


# watch -n 0.01 date +"%s.%N"
# fuser -k 8080/tcp

"""
timing of getting frame ofn separate thread
then between publishing and receiving
publishing shd be timechip with end of xyz frame to displayed time
"""

app = Flask(__name__)

@app.route("/test")
def test_route():
    return "Test working"

@app.route("/")
def index():
    return "/feed for the image; /feed/objDetect for the image with objDetect overlays"


def imgFromBytes(img_byte):
    try:
        jpg_original = base64.b64decode(img_byte)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        frame = cv2.imdecode(jpg_as_np, flags=1)
        if frame is None:
            raise ValueError("Failed to decode image")
        return frame
    except Exception as e:
        print(f"Error in imgFromBytes: {e}")
        return None

def gather_img_objDetect():
    try:
        stub = pb2_grpc.GreeterStub(grpc.insecure_channel("localhost:" + str(CFG["CAMERA_PORT"])))
    except Exception as e:
        print(f"Failed to create gRPC stub: {e}")
        return
    
    objDetect_stub = od_pb2_grpc.objDetectorStub(grpc.insecure_channel("localhost:" + str(CFG["OBJDETECT_PORT"])))
    
    while True:
        response = stub.get_rframe(pb2.HelloRequest(name="y"))  # in bytes

        # Process object detection
        results = objDetect_stub.askInfer_objDetect(od_pb2.objDetect_pic(name=response.message))
        frame = imgFromBytes(results.message)
        
        '''
        if CFG["CENT"] == 1 or CFG["CENT"] == 0:  # mqtt and get depth
            response = stub.get_depth(pb2.HelloRequest(name="y"))
            depth = response.img_data.decode("ascii")
            client.publish("i_depth", depth)
        '''
        _, frame = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame.tobytes() + b"\r\n")
    
    '''
    try:
        print("at least here")
        stub = pb2_grpc.GreeterStub(grpc.insecure_channel("localhost:" + str(CFG["CAMERA_PORT"])))
        #objDetect_stub = od_pb2_grpc.objDetectorStub(grpc.insecure_channel("localhost:" + str(CFG["OBJDETECT_PORT"])))



        while True:
            try:
                # Get frame and metadata from MISB video
                frame_response = stub.get_rframe(pb2.HelloRequest(name="y"))
                metadata_response = stub.get_metadata(pb2.HelloRequest(name="y"))
                
                # Process object detection
                #results = objDetect_stub.askInfer_objDetect(od_pb2.objDetect_pic(name=frame_response.message))
                #frame = imgFromBytes(results.message)
                frame = imgFromBytes(frame_response)

                
                # Extract metadata and add to buffer
                metadata = json.loads(metadata_response.message)
                metadata_buffer.append(metadata)

                # Get frame timestamp (assuming it's available in the frame_response)
                frame_timestamp = frame_response.timestamp  # Adjust this based on your actual implementation

                # Find the closest metadata entry
                closest_metadata = find_closest_metadata(frame_timestamp)
                
                # Add metadata overlay to frame
                frame = add_metadata_overlay(frame, closest_metadata)
                

                # Add object detection bounding boxes
                # frame = add_object_detection_overlay(frame, results)

                # Encode frame to JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()

                # Yield frame for Flask streaming
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

            except grpc.RpcError as e:
                print(f"RPC failed: {e}")
                time.sleep(1)  # Wait before retrying
    except Exception as e:
        print(f"Failed to create stub: {e}")
        print(traceback.format_exc())
        return "Error: gRPC server not available"
    '''

def find_closest_metadata(frame_timestamp):
    # Find the metadata entry with the closest timestamp to the frame
    timestamps = [m['timestamp'] for m in metadata_buffer]
    index = bisect.bisect_left(timestamps, frame_timestamp)
    
    if index == 0:
        return metadata_buffer[0]
    if index == len(metadata_buffer):
        return metadata_buffer[-1]
    
    before = metadata_buffer[index - 1]
    after = metadata_buffer[index]
    
    if abs(frame_timestamp - before['timestamp']) < abs(frame_timestamp - after['timestamp']):
        return before
    else:
        return after

def add_metadata_overlay(frame, metadata):
    # Add relevant metadata to the frame
    cv2.putText(frame, f"Timestamp: {metadata.get('timestamp', 'N/A')}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Lat: {metadata.get('latitude', 'N/A')}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Lon: {metadata.get('longitude', 'N/A')}", (10, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

def add_object_detection_overlay(frame, results):
    # Add bounding boxes and labels for detected objects
    for detection in results.detections:
        x_low, y_low = int(detection.x_low), int(detection.y_low)
        x_high, y_high = int(detection.x_high), int(detection.y_high)
        cv2.rectangle(frame, (x_low, y_low), (x_high, y_high), (0, 255, 0), 2)
        cv2.putText(frame, detection.class_name, (x_low, y_low - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

@app.route("/objDetect")
def flask_objDetect():
    return Response(gather_img_objDetect(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/kill")
def flask_kill():
    client.loop_stop()
    client.disconnect()
    return "yahoo"

def gather_img_feed():
    global sent_intrin
    print("Starting gather_img_feed")
    stub = pb2_grpc.GreeterStub(grpc.insecure_channel("localhost:" + str(CFG["CAMERA_PORT"])))
    print(f"Created gRPC stub for camera on port {CFG['CAMERA_PORT']}")

    sent_intrin = True

    while True:
        try:
            response = stub.get_rframe(pb2.HelloRequest(name="y"))  # in bytes

            frame = imgFromBytes(response.message)

            if frame is None:
                continue

            success, encoded_frame = cv2.imencode(".jpg", frame)
            if not success:
                continue

            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded_frame.tobytes() + b"\r\n")
        except Exception as e:
            print(f"Error in gather_img_feed: {e}")
            time.sleep(1)  # Add a delay before retrying

@app.route("/feed")
def flask_feed():
    # if CFG['DISP'] == 1 or CFG['CENT'] == 1:
    return Response(gather_img_feed(), mimetype="multipart/x-mixed-replace; boundary=frame")

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



class KlvEncoder:
    def __init__(self):
        self.klv_frame_counter = 0

    def encode_klv(self, metadata):
        klv_data = bytearray()

        # Timestamp (microseconds since 1970-01-01 00:00:00 UTC)
        key = b'\x02'
        value = common.str_to_bytes(metadata.items[5].string_value)
        print(f"metadataitems: {metadata.items[5]}")
        length = common.ber_encode(len(value))
        klv_data.extend(key + length + value)
        print(f"key: {key}")
        print(f"value: {value}")
        print(f"length: {length}")
        print("\n")

        '''
        # Latitude
        if 'latitude' in metadata:
            key = b'\x0D'
            value = common.float_to_bytes(metadata['latitude'], (-90, 90), (-(2**31), 2**31 - 1))
            length = common.ber_encode(len(value))
            klv_data.extend(key + length + value)

        # Longitude
        if 'longitude' in metadata:
            key = b'\x0E'
            value = common.float_to_bytes(metadata['longitude'], (-180, 180), (-(2**31), 2**31 - 1))
            length = common.ber_encode(len(value))
            klv_data.extend(key + length + value)

        # Altitude
        if 'altitude' in metadata:
            key = b'\x0F'
            value = common.float_to_bytes(metadata['altitude'], (-900, 19000), (0, 2**16 - 1))
            length = common.ber_encode(len(value))
            klv_data.extend(key + length + value)

        # Platform Heading Angle
        if 'heading' in metadata:
            key = b'\x05'
            value = common.float_to_bytes(metadata['heading'], (0, 360), (0, 2**16 - 1))
            length = common.ber_encode(len(value))
            klv_data.extend(key + length + value)

        # Platform Pitch Angle
        if 'pitch' in metadata:
            key = b'\x06'
            value = common.float_to_bytes(metadata['pitch'], (-20, 20), (-(2**15), 2**15 - 1))
            length = common.ber_encode(len(value))
            klv_data.extend(key + length + value)

        # Platform Roll Angle
        if 'roll' in metadata:
            key = b'\x07'
            value = common.float_to_bytes(metadata['roll'], (-50, 50), (-(2**15), 2**15 - 1))
            length = common.ber_encode(len(value))
            klv_data.extend(key + length + value)

        # Add more metadata fields as needed
        '''
        
        # Wrap the KLV data in a SMPTE ST 336 key
        universal_set_key = b'\x06\x0E\x2B\x34\x02\x0B\x01\x01\x0E\x01\x03\x01\x01\x00\x00\x00'
        universal_set_length = common.ber_encode(len(klv_data))
        
        full_klv = universal_set_key + universal_set_length + klv_data

        self.klv_frame_counter += 1

        return full_klv

def create_misb_stream(frame_queue, metadata_queue):
    udp_output = 'udp://127.0.0.1:5555'  # Change this to your desired output
    
    try:
        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='1920x1080', framerate=30)
            .output(udp_output, format='mpegts', vcodec='libx264', preset='ultrafast')
            .global_args('-i', 'pipe:')  # Second input for KLV data
            .global_args('-map', '0:v', '-map', '0:d')  # Map video from first input, all streams from second input
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stderr=True)
        )
    except ffmpeg.Error as e:
        print(f"Error starting ffmpeg process: {e}")
        return

    # process = (
    #     ffmpeg
    #     .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='1920x1080', framerate=30)
    #     .output(udp_output, format='mpegts', vcodec='libx264', preset='ultrafast', tune='zerolatency',
    #             force_key_frames='expr:gte(t,0)',)
    #     .overwrite_output()
    #     .run_async(pipe_stdin=True)
    # )

    print("Starting MISB stream creation")
    import struct
    while True:
        try:
            frame = frame_queue.get(timeout=5)  # Increased timeout
            metadata = metadata_queue.get(timeout=5)  # Uncomment when metadata is implemented

            if frame is not None:
                # Write frame to ffmpeg process
                process.stdin.write(frame.tobytes())
                #print(frame.tobytes())

                if metadata is not None:
                    #process.stdin.write(metadata.metadataitem)
                    print(metadata.metadataitem)
                else:
                    print("Received None metadata")         ###TODO! WRITE NONE!
            else:
                print("Received None frame")

        except queue.Empty:
            print("Queue is empty, waiting for more frames")
            continue
        except Exception as e:
            print(f"Error in MISB stream creation: {e}")
            break
    
    process.stdin.close()
    process.wait()

def gather_frames_and_metadata(frame_queue, metadata_queue):
    stub = pb2_grpc.GreeterStub(grpc.insecure_channel("localhost:" + str(CFG["CAMERA_PORT"])))
    objDetect_stub = od_pb2_grpc.objDetectorStub(grpc.insecure_channel("localhost:" + str(CFG["OBJDETECT_PORT"])))

    print("Starting frame gathering")
    while True:
        try:
            frame_response = stub.get_rframe(pb2.HelloRequest(name="y"))
            metadata = stub.get_metadata(pb2.HelloRequest(name="y"))

            results = objDetect_stub.askInfer_objDetect(od_pb2.objDetect_pic(name=frame_response.message))
            frame = imgFromBytes(results.message)

            if frame is not None:
                frame_queue.put(frame)
            else:
                print("Failed to decode frame")
            metadata_queue.put(metadata)
        except Exception as e:
            print(f"Error gathering frames and metadata: {e}")
            time.sleep(1)  # Add a delay before retrying

@app.route("/MisbOut")
def misb_out():
    frame_queue = queue.Queue(maxsize=1)  # Limit queue size
    metadata_queue = queue.Queue(maxsize=1)

    # Start frame gathering thread
    threading.Thread(target=gather_frames_and_metadata, args=(frame_queue, metadata_queue), daemon=True).start()

    # Start MISB stream creation thread
    threading.Thread(target=create_misb_stream, args=(frame_queue, metadata_queue), daemon=True).start()

    return "MISB stream started. Connect to udp://127.0.0.1:5555 to view the stream."

#################################################GSTREAMER####################################################################

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject

GObject.threads_init()
Gst.init(None)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def create_gstreamer_pipeline(frame_queue, metadata_queue):
    # video source elements
    vsrc = Gst.ElementFactory.make("appsrc", "vidsrc")
    vqueue = Gst.ElementFactory.make("queue")
    vtee = Gst.ElementFactory.make("tee")

    # klv source elements
    appsrc = Gst.ElementFactory.make("appsrc")
    queue_klv = Gst.ElementFactory.make("queue")

    # # display elements
    # queue_display = Gst.ElementFactory.make("queue")
    # vcvt = Gst.ElementFactory.make("videoconvert", "vidcvt")
    # vsink = Gst.ElementFactory.make("autovideosink", "vidsink")

    # recording elements
    queue_record = Gst.ElementFactory.make("queue")
    vcvt_encoder = Gst.ElementFactory.make("videoconvert")
    encoder = Gst.ElementFactory.make("x264enc")
    muxer = Gst.ElementFactory.make("mpegtsmux")
    # filesink = Gst.ElementFactory.make("filesink")
    udpsink = Gst.ElementFactory.make("udpsink")

    # configure video elementvsrc = Gst.ElementFactory.make("appsrc", "vidsrc")
    vqueue = Gst.ElementFactory.make("queue")
    vtee = Gst.ElementFactory.make("tee")

    # klv source elements
    appsrc = Gst.ElementFactory.make("appsrc")
    queue_klv = Gst.ElementFactory.make("queue")

    # # display elements
    # queue_display = Gst.ElementFactory.make("queue")
    # vcvt = Gst.ElementFactory.make("videoconvert", "vidcvt")
    # vsink = Gst.ElementFactory.make("autovideosink", "vidsink")

    vid_width = 1920
    vid_height = 1080
    vid_frame_rate = 30
    caps_str = "video/x-raw"
    caps_str += ",format=(string)RGB,width={},height={}".format(vid_width,vid_height)
    caps_str += ",framerate={}/1".format(int(vid_frame_rate))
    vcaps = Gst.Caps.from_string(caps_str)
    vsrc.set_property("caps", vcaps);
    vsrc.set_property("format", Gst.Format.TIME)

    # configure appsrc element
    caps_str = "meta/x-klv"
    caps_str += ",parsed=True"
    caps = Gst.Caps.from_string(caps_str)
    appsrc.set_property("caps", caps)
    # appsrc.connect("need-data", klv_need_data)
    appsrc.set_property("format", Gst.Format.TIME)

    # configure encoder
    encoder.set_property("noise-reduction", 1000)
    encoder.set_property("threads", 4)
    encoder.set_property("bitrate", 1755)
    encoder.set_property("tune", "zerolatency")

    # configure filesink
    # out_file = "output3.ts"
    # filesink.set_property("location", out_file)
    # filesink.set_property("async", 0)

    # configure udpsink
    udpsink.set_property("host", "127.0.0.1")
    udpsink.set_property("port", 5555)
    udpsink.set_property("async", 0)

    pipeline = Gst.Pipeline()
    pipeline.add(vsrc)
    pipeline.add(vqueue)
    pipeline.add(vtee)
    pipeline.add(appsrc)
    pipeline.add(queue_klv)
    # pipeline.add(queue_display)
    # pipeline.add(vcvt)
    # pipeline.add(vsink)
    pipeline.add(queue_record)
    pipeline.add(vcvt_encoder)
    pipeline.add(encoder)
    pipeline.add(muxer)
    pipeline.add(udpsink)

    # link video elements
    vsrc.link(vqueue)
    vqueue.link(vtee)

    # # link display elements
    # vtee.link(queue_display)
    # queue_display.link(vcvt)
    # vcvt.link(vsink)

    # link recording elements
    vtee.link(queue_record)
    queue_record.link(vcvt_encoder)
    vcvt_encoder.link(encoder)
    encoder.link(muxer)
    muxer.link(udpsink)

    # link klv elements
    appsrc.link(queue_klv)
    queue_klv.link(muxer)

    pipeline.set_state(Gst.State.PLAYING)

    klv_frame_rate = 30
    
    metadata = metadata_queue.get(timeout=5).metadataitem
    timestamp = 0

    klv_done = False
    vid_done = False
    vid_frame_counter = 0
    klv_frame_counter = 0

    t = 0
    last_klv_t = 0
    last_vid_t = 0
    while True:
        if vid_done and klv_done:
            break
        if t - last_klv_t >= 1.0 / klv_frame_rate:
            if not klv_done:
                klv_bytes = metadata
                if klv_bytes:
                    klvbuf = Gst.Buffer.new_allocate(None, 188, None)
                    klvbuf.fill(0, klv_bytes)
                    klvbuf.pts = int(t * 1e9)
                    klvbuf.dts = int(t * 1e9)

                    appsrc.emit("push-buffer", klvbuf)
                    klv_frame_counter += 1
                    last_klv_t = t
                    print("klv {} {}".format(klv_frame_counter, last_klv_t))
                else:
                    klv_done = True

        if t - last_vid_t >= 1.0 / vid_frame_rate:
            if not vid_done:
                img_byte = frame_queue.get(timeout=5)
                logging.debug("Retrieved frame from queue, queue size: %d", frame_queue.qsize())
                if img_byte:
                    jpg_original = base64.b64decode(img_byte)
                    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
                    frame = cv2.imdecode(jpg_as_np, flags=1)
                    
                    # Convert BGR to RGB
                    frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    data = frame_RGB.tostring()
                    
                    vidbuf = Gst.Buffer.new_allocate(None, len(data), None)
                    vidbuf.fill(0, data)
                    vidbuf.pts = int(t * 1e9)
                    vidbuf.dts = int(t * 1e9)

                    vsrc.emit("push-buffer", vidbuf)
                    vid_frame_counter += 1
                    last_vid_t = t
                    print("vid {} {}".format(vid_frame_counter, last_vid_t))
                else:
                    vid_done = True
                    continue

        t += 0.000001
        #print(t)

    vsrc.emit("end-of-stream")
    appsrc.emit("end-of-stream")

    bus = pipeline.get_bus()

    while True:
        msg = bus.poll(Gst.MessageType.ANY, Gst.CLOCK_TIME_NONE)
        t = msg.type
        if t == Gst.MessageType.EOS:
            print("EOS")
            break
            pipeline.set_state(Gst.State.NULL)
        elif t == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            # print("Error: %s" % err, debug)
            break
        elif t == Gst.MessageType.WARNING:
            err, debug = msg.parse_warning()
            # print("Warning: %s" % err, debug)
        elif t == Gst.MessageType.STATE_CHANGED:
            pass
        elif t == Gst.MessageType.STREAM_STATUS:
            pass
        else:
            pass
            # print(t)
            # print("Unknown message: %s" % msg)

    pipeline.set_state(Gst.State.NULL)

    print("Bye")

def gather_frames_and_metadata(frame_queue, metadata_queue):
    stub = pb2_grpc.GreeterStub(grpc.insecure_channel("localhost:" + str(CFG["CAMERA_PORT"])))
    objDetect_stub = od_pb2_grpc.objDetectorStub(grpc.insecure_channel("localhost:" + str(CFG["OBJDETECT_PORT"])))
    
    print("Starting frame gathering")
    while True:
        try:
            frame_response = stub.get_rframe(pb2.HelloRequest(name="y"))
            metadata = stub.get_metadata(pb2.HelloRequest(name="y"))
            results = objDetect_stub.askInfer_objDetect(od_pb2.objDetect_pic(name=frame_response.message))
            frame = results.message
            

            if frame is not None:
                if frame_queue.full():
                    logging.debug("Frame queue is full, removing the oldest frame.")
                    frame_queue.get()  # Remove the first frame in the queue
                frame_queue.put(frame)
                logging.debug("Added frame to queue, queue size: %d", frame_queue.qsize())
            else:
                print("Failed to decode frame")
            
            metadata_queue.put(metadata)
            
        except Exception as e:
            print(f"Error gathering frames and metadata: {e}")
            time.sleep(1)

@app.route("/GstreamerOut")
def gstreamer_out():
    frame_queue = queue.Queue(maxsize=200)
    metadata_queue = queue.Queue(maxsize=200)
    
    threading.Thread(target=gather_frames_and_metadata, args=(frame_queue, metadata_queue), daemon=True).start()
    threading.Thread(target=create_gstreamer_pipeline, args=(frame_queue, metadata_queue), daemon=True).start()
    
    return "GStreamer pipeline started. Connect to udp://127.0.0.1:5555 to view the stream."

def kill_process_on_port(port):
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            try:
                process = psutil.Process(conn.pid)
                process.terminate()
                print(f"Terminated process {conn.pid} using port {port}")
            except psutil.NoSuchProcess:
                pass

if __name__ == "__main__":

    def on_connect(client, userdata, flags, rc):
        print("Connected to MQTT broker")

    yaml_source = os.path.join(os.path.dirname(__file__), "CONFIG.yml")

    with Path(yaml_source).open() as f:
        CFG = yaml.load(f, Loader=yaml.Loader)


    port = int(CFG["CAMERA_PORT"])
    kill_process_on_port(port)
    time.sleep(0.5)  # if not the app.run runs first and u die

    client = mqtt.Client()
    client.on_connect = on_connect
    client.connect(str(CFG["MQTT_HOST"]), int(CFG["MQTT_PORT"]))

    client.loop_start()

    logging.basicConfig()

    time.sleep(1)  # if not the app.run runs first and u die
    app.run(host="0.0.0.0", port=CFG["CLIENT_PORT"], threaded=True)


# the remnants of hell
"""
The final braincell
aim: find out what dies
theories: port, mqtt, something in grpc

restart
<no intrinsics calling, only depth frame and normal frame calling>
- runs normal - works
- mqtt disconnect/end, fuser kill client - works
- mqtt disconnect/end, fuser kill client - 1st grpc function call, lags..... works
- mqtt disconnect.end, fuser kill - client AND server,  - 1st grpc function call, lags.. works
- fuser kill client AND server - works 
- nothing -1st grpc function call, lags....works
- nothing - port 5000 (client) in use - kill port - work
- nothing - it printed but before even pressing enter - lag.. - work

restart
<do nothing and just keep running and running and running>
- working even before pressing enter
- client port in use - restarted client port - 1st grpc function call, lags...timeout....reload, works
- nothing - not even at 1st grpc - lag.............. - reloaded, at 1st grpc, lags... .works
- force kill client - 1st grpc function call...lag ... timeout - reload works
- nothing - 1st grpc function call ....lag ....timepout - reload ... 1st call ... lag. ...timeout ....
- killed server ports (many processes), and client port - works
- killed server and client ports - works 
- killed again - works??????
- made get intrin exist (not called)
- killed server and client ports - works
- calling intrin with hardcoded integers again, not using the integers
- kill both again: IT WORK
- bring back thge get_intrin
- work
- bring back get intrin fully



> ports not ending processes is probably a problem to some extent
> mqtt probably doesn't affect
> the grpc is still the main problem (?)
"""