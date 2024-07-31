from __future__ import print_function

import logging

import grpc
import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the current script
sys.path.append(os.path.join(dir_path, "../protobufs"))
import main_server_pb2 as pb2
import main_server_pb2_grpc as pb2_grpc


import objDetect_pb2_grpc
import objDetect_pb2

import poseEst_pb2
import poseEst_pb2_grpc

import newMOT_pb2
import newMOT_pb2_grpc

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

prevtime = time.time()
t = time.time()


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


@app.route("/")
def index():
    return "welcome to mcdonalds :)"


def imgFromBytes(img_byte):
    jpg_original = base64.b64decode(img_byte)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    frame = cv2.imdecode(jpg_as_np, flags=1)
    return frame


def imgToBytes(img):
    _, buffer = cv2.imencode(".jpg", img)
    data = base64.b64encode(buffer)
    return data


def gather_img_objDetect():
    stub = pb2_grpc.GreeterStub(grpc.insecure_channel("localhost:" + str(CFG["CAMERA_PORT"])))
    objDetect_stub = objDetect_pb2_grpc.objDetectorStub(
        grpc.insecure_channel("localhost:" + str(CFG["OBJDETECT_PORT"]))
    )

    # jn test
    x = 0
    y = 0
    z = 0
    id = 0

    cropped_bytes = 0
    exist_bytes = False

    while True:
        response = stub.get_rframe(pb2.HelloRequest(name="y"))  # in bytes
        results = objDetect_stub.askInfer_objDetect(
            objDetect_pb2.objDetect_pic(name=response.message)
        )  # infer
        frame = imgFromBytes(results.message)

        # jn test
        x_low = round(float(results.x_lo))
        y_low = round(float(results.y_lo))
        x_high = round(float(results.x_hi))
        y_high = round(float(results.y_hi))

        x_mid = round(float((x_low + x_high) / 2))
        y_mid = round(float((y_low + y_high) / 2))

        coords = stub.ask_xyz(pb2.xyz(x=str(x_mid), y=str(y_mid)))

        # if x_low != x_high and y_low != y_high:
        #     cropped = frame[y_low:y_high, x_low:x_high]
        #     h,w,z = cropped.shape
        #     cropped = cv2.resize(cropped,(int(w/2), int(h/2)))

        #     _, buffer = cv2.imencode('.jpg', cropped)
        #     data = base64.b64encode(buffer)
        #     cropped_bytes = data.decode('ascii')

        if (
            coords.x != str(-1) and coords.y != str(-1) and coords.z != (-1)
        ):  # trash for if no depth val at pixel
            if coords.x != 0 or coords.y != 0 or coords.z != 0:
                x = coords.x
                y = coords.y
                z = coords.z

        label = "forward:" + str(x) + " right: " + str(y) + " down: " + str(z)
        frame = cv2.putText(
            frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA
        )

        if CFG["CENT"] == 0:
            # client.publish("x", x)
            # client.publish("y", y)
            # client.publish("z", z)
            # if exist_bytes:
            #     client.publish("cropped_img", cropped_bytes) #bytesalrstring

            j_i = {"x": str(x), "y": str(y), "z": str(z), "chip": str(cropped_bytes)}
            data_out = json.dumps(j_i)  # encode
            # do sth data_out
            # with open("sample.json", "w") as outfile:
            #     json.dump(j_i, outfile)
            # with open("samplemax.json", "w") as outfile:
            #     print("io")
            #     _, buffer = cv2.imencode('.jpg', frame)
            #     data = base64.b64encode(buffer)
            #     cropped_bytes = data.decode('ascii')
            #     j_i = {'x': str(x), 'y': str(y), 'z': str(z), 'chip': str(cropped_bytes)}
            #     json.dump(j_i, outfile)

            client.publish("xyz_chip", data_out)

        _, frame = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame.tobytes() + b"\r\n")


@app.route("/feed/objDetect")
def flask_objDetect():
    return Response(gather_img_objDetect(), mimetype="multipart/x-mixed-replace; boundary=frame")


def gather_img_poseEst():
    stub = pb2_grpc.GreeterStub(grpc.insecure_channel("localhost:" + str(CFG["CAMERA_PORT"])))
    poseEst_stub = poseEst_pb2_grpc.poseEstorStub(
        grpc.insecure_channel("localhost:" + str(CFG["POSEEST_PORT"]))
    )

    while True:
        response = stub.get_rframe(pb2.HelloRequest(name="y"))  # in bytes
        results = poseEst_stub.askInfer_poseEst(
            poseEst_pb2.poseEst_pic(name=response.message)
        )  # infer #here
        frame = imgFromBytes(results.message)
        _, frame = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame.tobytes() + b"\r\n")


@app.route("/feed/poseEst")
def flask_poseEst():
    return Response(gather_img_poseEst(), mimetype="multipart/x-mixed-replace; boundary=frame")


def gather_img_MOT(plot_in_rviz):
    """Gather Images from MOT algo service and returns images of detected classes with bounding boxes

    Args:
        plot_in_rviz (Boolean): Choose whether you want to simulate robot in rviz

    Yields:
        Image Bytes
    """
    stub = pb2_grpc.GreeterStub(grpc.insecure_channel("localhost:" + str(CFG["CAMERA_PORT"])))
    MOT_stub = newMOT_pb2_grpc.MOTorStub(grpc.insecure_channel("localhost:" + str(CFG["MOT_PORT"])))
    id = 0
    cropped_bytes = 0
    exist_bytes = False

    while True:
        x = 0
        y = 0
        z = 0

        xmap = 0
        ymap = 0
        zmap = 0

        detect = False
        response = stub.get_rframe(pb2.HelloRequest(name="y"))  # in bytes
        timestamp("x")
        results = MOT_stub.askInfer_MOT(newMOT_pb2.MOT_pic(image=response.message))  # infer #here
        timestamp("ask infer:")

        frame = imgFromBytes(results.image)
        first_detection = results.detections
        try:
            first_detection = results.detections[0]
        except:
            pass

        try:
            x_low = round(float(first_detection.x_low))
            y_low = round(float(first_detection.y_low))
            x_high = round(float(first_detection.x_high))
            y_high = round(float(first_detection.y_high))

            x_mid = round(float((x_low + x_high) / 2))
            y_mid = round(float((y_low + y_high) / 2))

            print(x_mid, y_mid, type(x_mid), "xmid ymid")  # (0,0,int,qweqwe)

            # timestamp("x")
            # print(type(coords.x), coords.y, coords.z)  #(string,-1,-1)
            # print(str(coords.x),'rrrr')
            # coords_map=stub.ask_xyzmap(pb2.xyzmap(xnew = str(0), ynew=str(0), znew=str(0)))
            # print(coords_map,'coords_map')
            timestamp("ask_xyz time")

            if x_low != x_high and y_low != y_high:
                cropped = frame[y_low:y_high, x_low:x_high]
                h, w, z = cropped.shape
                cropped = cv2.resize(cropped, (int(w / 2), int(h / 2)))

                _, buffer = cv2.imencode(".jpg", cropped)
                data = base64.b64encode(buffer)
                cropped_bytes = data.decode("ascii")

            timestamp("crop")
            if x_mid != 0 and y_mid != 0:  # if detect something
                coords = stub.ask_xyz(pb2.xyz(x=str(x_mid), y=str(y_mid)))  # xyz POV of vehicle
                if (
                    coords.x != str(-1) and coords.y != str(-1) and coords.z != (-1)
                ):  # its -1 -1 -1 if camera face table
                    if coords.x != 0 or coords.y != 0 or coords.z != 0:
                        x = coords.x
                        y = coords.y
                        z = coords.z
                        print(
                            x, y, z, "test6"
                        )  # wont print this if camera face table, but will print if facing wall

                        if plot_in_rviz == True:
                            coords_map = stub.ask_xyzmap(
                                pb2.xyzmap(
                                    xnew=str(coords.x), ynew=str(coords.y), znew=str(coords.z)
                                )
                            )  # xyz POV of map
                            xmap = coords_map.xnew
                            ymap = coords_map.ynew
                            zmap = coords_map.znew

            COLOR = (0, 255, 0)  # GREEN
            LINE_THICKNESS = 2
            frame = cv2.circle(
                frame, (int(x_mid), int(y_mid)), radius=10, color=(0, 0, 255), thickness=-1
            )
            cv2.rectangle(frame, (x_low, y_low), (x_high, y_high), COLOR, LINE_THICKNESS)
        except:
            pass

            # print(x,y,z), forward right down is perspective of camera, stub.ask_xyz returns converted values for Rviz already(object to vehicle)

        label = (
            "detected x: "
            + str(round(float(x), 3))
            + " y: "
            + str(round(float(y), 3))
            + " z: "
            + str(round(float(z), 3))
        )  # xyz POV of robot
        frame = cv2.putText(
            frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA
        )

        label_map = (
            "detected x: "
            + str(round(float(xmap), 3))
            + " y: "
            + str(round(float(ymap), 3))
            + " z: "
            + str(round(float(zmap), 3))
        )  # xyz on map
        frame = cv2.putText(
            frame, label_map, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA
        )

        if CFG["CENT"] == 0:
            # client.publish("x", x)
            # client.publish("y", y)
            # client.publish("z", z)
            # if exist_bytes:
            #     client.publish("cropped_img", cropped_bytes) #bytesalrstring

            if x != 0 and y != 0 and z != 0:
                j_i = {"x": str(x), "y": str(y), "z": str(z), "chip": str(cropped_bytes)}
                data_out = json.dumps(j_i)  # encode

                j_i_map = {
                    "x": str(xmap),
                    "y": str(ymap),
                    "z": str(zmap),
                    "chip": str(cropped_bytes),
                }
                data_out_map = json.dumps(j_i_map)  # encode

                # do sth data_out
                # with open("sample.json", "w") as outfile:
                #     json.dump(j_i, outfile)
                # with open("samplemax.json", "w") as outfile:
                #     print("io")
                #     _, buffer = cv2.imencode('.jpg', frame)
                #     data = base64.b64encode(buffer)
                #     cropped_bytes = data.decode('ascii')
                #     j_i = {'x': str(x), 'y': str(y), 'z': str(z), 'chip': str(cropped_bytes)}
                #     json.dump(j_i, outfile)

                client.publish("xyz_chip", data_out)  # MQTT
                # client.publish("xyz_map", data_out_map) #MQTT for map

        timestamp("time to publish")

        # if CFG['DISP'] == 1:
        _, frame = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame.tobytes() + b"\r\n")
        # else:
        #     pass
        #     # yield("wahoo")


def gather_img_MOT(plot_in_rviz):
    """Gather Images from MOT algo service and returns images of detected classes with bounding boxes

    Args:
        plot_in_rviz (Boolean): Choose whether you want to simulate robot in rviz

    Yields:
        Image Bytes
    """
    stub = pb2_grpc.GreeterStub(grpc.insecure_channel("localhost:" + str(CFG["CAMERA_PORT"])))
    MOT_stub = newMOT_pb2_grpc.MOTorStub(grpc.insecure_channel("localhost:" + str(CFG["MOT_PORT"])))
    id = 0
    cropped_bytes = 0
    exist_bytes = False

    while True:
        x = 0
        y = 0
        z = 0

        xmap = 0
        ymap = 0
        zmap = 0

        detect = False
        response = stub.get_rframe(pb2.HelloRequest(name="y"))  # in bytes
        timestamp("x")
        results = MOT_stub.askInfer_MOT(newMOT_pb2.MOT_pic(image=response.message))  # infer #here
        timestamp("ask infer:")

        frame = imgFromBytes(results.image)
        first_detection = results.detections
        try:
            first_detection = results.detections[0]
        except:
            pass

        try:
            x_low = round(float(first_detection.x_low))
            y_low = round(float(first_detection.y_low))
            x_high = round(float(first_detection.x_high))
            y_high = round(float(first_detection.y_high))

            x_mid = round(float((x_low + x_high) / 2))
            y_mid = round(float((y_low + y_high) / 2))

            print(x_mid, y_mid, type(x_mid), "xmid ymid")  # (0,0,int,qweqwe)

            # timestamp("x")
            # print(type(coords.x), coords.y, coords.z)  #(string,-1,-1)
            # print(str(coords.x),'rrrr')
            # coords_map=stub.ask_xyzmap(pb2.xyzmap(xnew = str(0), ynew=str(0), znew=str(0)))
            # print(coords_map,'coords_map')
            timestamp("ask_xyz time")

            if x_low != x_high and y_low != y_high:
                cropped = frame[y_low:y_high, x_low:x_high]
                h, w, z = cropped.shape
                cropped = cv2.resize(cropped, (int(w / 2), int(h / 2)))

                _, buffer = cv2.imencode(".jpg", cropped)
                data = base64.b64encode(buffer)
                cropped_bytes = data.decode("ascii")

            timestamp("crop")
            if x_mid != 0 and y_mid != 0:  # if detect something
                coords = stub.ask_xyz(pb2.xyz(x=str(x_mid), y=str(y_mid)))  # xyz POV of vehicle
                if (
                    coords.x != str(-1) and coords.y != str(-1) and coords.z != (-1)
                ):  # its -1 -1 -1 if camera face table
                    if coords.x != 0 or coords.y != 0 or coords.z != 0:
                        x = coords.x
                        y = coords.y
                        z = coords.z
                        print(
                            x, y, z, "test6"
                        )  # wont print this if camera face table, but will print if facing wall

                        if plot_in_rviz == True:
                            coords_map = stub.ask_xyzmap(
                                pb2.xyzmap(
                                    xnew=str(coords.x), ynew=str(coords.y), znew=str(coords.z)
                                )
                            )  # xyz POV of map
                            xmap = coords_map.xnew
                            ymap = coords_map.ynew
                            zmap = coords_map.znew

            COLOR = (0, 255, 0)  # GREEN
            LINE_THICKNESS = 2
            frame = cv2.circle(
                frame, (int(x_mid), int(y_mid)), radius=10, color=(0, 0, 255), thickness=-1
            )
            cv2.rectangle(frame, (x_low, y_low), (x_high, y_high), COLOR, LINE_THICKNESS)
        except:
            pass

            # print(x,y,z), forward right down is perspective of camera, stub.ask_xyz returns converted values for Rviz already(object to vehicle)

        label = "vehicle x: " + str(x) + " y: " + str(y) + " z: " + str(z)  # xyz POV of vehicle
        frame = cv2.putText(
            frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA
        )

        label_map = "map x:" + str(xmap) + " y: " + str(ymap) + " z: " + str(zmap)
        frame = cv2.putText(
            frame, label_map, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA
        )

        if CFG["CENT"] == 0:
            # client.publish("x", x)
            # client.publish("y", y)
            # client.publish("z", z)
            # if exist_bytes:
            #     client.publish("cropped_img", cropped_bytes) #bytesalrstring

            if x != 0 and y != 0 and z != 0:
                j_i = {"x": str(x), "y": str(y), "z": str(z), "chip": str(cropped_bytes)}
                data_out = json.dumps(j_i)  # encode

                j_i_map = {
                    "x": str(xmap),
                    "y": str(ymap),
                    "z": str(zmap),
                    "chip": str(cropped_bytes),
                }
                data_out_map = json.dumps(j_i_map)  # encode

                # do sth data_out
                # with open("sample.json", "w") as outfile:
                #     json.dump(j_i, outfile)
                # with open("samplemax.json", "w") as outfile:
                #     print("io")
                #     _, buffer = cv2.imencode('.jpg', frame)
                #     data = base64.b64encode(buffer)
                #     cropped_bytes = data.decode('ascii')
                #     j_i = {'x': str(x), 'y': str(y), 'z': str(z), 'chip': str(cropped_bytes)}
                #     json.dump(j_i, outfile)

                client.publish("xyz_chip", data_out)  # MQTT
                # client.publish("xyz_map", data_out_map) #MQTT for map

        timestamp("time to publish")

        # if CFG['DISP'] == 1:
        _, frame = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame.tobytes() + b"\r\n")
        # else:
        #     pass
        #     # yield("wahoo")


@app.route("/feed/mot")
def flask_MOT():
    plot_in_rviz = False
    # if CFG['DISP'] == 0:
    # gather_img_MOT()
    # return "MOT_nodisplay" #Response(gather_img_MOT(), mimetype='multipart/x-mixed-replace; boundary=frame')

    # elif CFG['DISP'] == 1: #1 dev
    return Response(
        gather_img_MOT(plot_in_rviz), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def gather_img_MOT_infrared(plot_in_rviz):
    """Gather Images from MOT algo service and returns images of detected classes with bounding boxes

    Args:
        plot_in_rviz (Boolean): Choose whether you want to simulate robot in rviz

    Yields:
        Image Bytes
    """
    stub = pb2_grpc.GreeterStub(grpc.insecure_channel("localhost:" + str(CFG["CAMERA_PORT"])))
    MOT_stub = newMOT_pb2_grpc.MOTorStub(grpc.insecure_channel("localhost:" + str(CFG["MOT_PORT"])))
    id = 0
    cropped_bytes = 0
    exist_bytes = False

    while True:
        x = 0
        y = 0
        z = 0

        xmap = 0
        ymap = 0
        zmap = 0

        detect = False
        response = stub.get_infrared_frame(pb2.HelloRequest(name="y"))  # in bytes
        timestamp("x")
        results = MOT_stub.askInfer_MOT(newMOT_pb2.MOT_pic(image=response.message))  # infer #here
        timestamp("ask infer:")

        frame = imgFromBytes(results.image)
        first_detection = results.detections
        try:
            first_detection = results.detections[0]
        except:
            pass

        try:
            x_low = round(float(first_detection.x_low))
            y_low = round(float(first_detection.y_low))
            x_high = round(float(first_detection.x_high))
            y_high = round(float(first_detection.y_high))

            x_mid = round(float((x_low + x_high) / 2))
            y_mid = round(float((y_low + y_high) / 2))

            print(x_mid, y_mid, type(x_mid), "xmid ymid")  # (0,0,int,qweqwe)
            timestamp("ask_xyz time")

            if x_low != x_high and y_low != y_high:
                cropped = frame[y_low:y_high, x_low:x_high]
                h, w, z = cropped.shape
                cropped = cv2.resize(cropped, (int(w / 2), int(h / 2)))

                _, buffer = cv2.imencode(".jpg", cropped)
                data = base64.b64encode(buffer)
                cropped_bytes = data.decode("ascii")

            timestamp("crop")
            if x_mid != 0 and y_mid != 0:  # if detect something
                coords = stub.ask_xyz(pb2.xyz(x=str(x_mid), y=str(y_mid)))  # xyz POV of vehicle
                if (
                    coords.x != str(-1) and coords.y != str(-1) and coords.z != (-1)
                ):  # its -1 -1 -1 if camera face table
                    if coords.x != 0 or coords.y != 0 or coords.z != 0:
                        x = coords.x
                        y = coords.y
                        z = coords.z
                        print(
                            x, y, z, "test6"
                        )  # wont print this if camera face table, but will print if facing wall

                        if plot_in_rviz == True:
                            coords_map = stub.ask_xyzmap(
                                pb2.xyzmap(
                                    xnew=str(coords.x), ynew=str(coords.y), znew=str(coords.z)
                                )
                            )  # xyz POV of map
                            xmap = coords_map.xnew
                            ymap = coords_map.ynew
                            zmap = coords_map.znew

            COLOR = (0, 255, 0)  # GREEN
            LINE_THICKNESS = 2
            frame = cv2.circle(
                frame, (int(x_mid), int(y_mid)), radius=10, color=(0, 0, 255), thickness=-1
            )
            cv2.rectangle(frame, (x_low, y_low), (x_high, y_high), COLOR, LINE_THICKNESS)
        except:
            pass
        label = "vehicle x: " + str(x) + " y: " + str(y) + " z: " + str(z)  # xyz POV of vehicle
        frame = cv2.putText(
            frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA
        )

        label_map = "map x:" + str(xmap) + " y: " + str(ymap) + " z: " + str(zmap)
        frame = cv2.putText(
            frame, label_map, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA
        )

        if CFG["CENT"] == 0:
            if x != 0 and y != 0 and z != 0:
                j_i = {"x": str(x), "y": str(y), "z": str(z), "chip": str(cropped_bytes)}
                data_out = json.dumps(j_i)  # encode

                j_i_map = {
                    "x": str(xmap),
                    "y": str(ymap),
                    "z": str(zmap),
                    "chip": str(cropped_bytes),
                }
                data_out_map = json.dumps(j_i_map)  # encode
                client.publish("xyz_chip", data_out)  # MQTT

        timestamp("time to publish")

        _, frame = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame.tobytes() + b"\r\n")


@app.route("/feed/infrared/mot")
def flask_infrared_MOT():
    plot_in_rviz = False
    return Response(
        gather_img_MOT_infrared(plot_in_rviz), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def gather_img_infrared():
    global sent_intrin
    stub = pb2_grpc.GreeterStub(grpc.insecure_channel("localhost:" + str(CFG["CAMERA_PORT"])))

    if (CFG["CENT"] == 1 or CFG["CENT"] == 0) and not sent_intrin:  # mqtt intrinsics
        i = stub.get_intrin(pb2.HelloRequest(name="y"))  # seems to work, to test in MOT later
        i_coeffs = i.coeffs.decode("ascii")

        j_i = {
            "width": i.width,
            "height": i.height,
            "ppx": i.ppx,
            "ppy": i.ppy,
            "fx": i.fx,
            "fy": i.fy,
            "coeffs": i_coeffs,
        }
        data_out = json.dumps(j_i)  # encode
        client.publish("i", data_out, retain=True)

        sent_intrin = True

    while True:
        response = stub.get_infrared_frame(pb2.HelloRequest(name="y"))  # in bytes
        frame = imgFromBytes(response.message)

        if CFG["CENT"] == 1 or CFG["CENT"] == 0:  # mqtt and get depth
            response = stub.get_depth(pb2.HelloRequest(name="y"))
            depth = response.img_data.decode("ascii")
            client.publish("i_depth", depth)

        _, frame = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame.tobytes() + b"\r\n")


@app.route("/feed/infrared")
def flask_infrared_detection():
    return Response(gather_img_infrared(), mimetype="multipart/x-mixed-replace; boundary=frame")


sent_intrin = False


def gather_img_feed():
    global sent_intrin
    stub = pb2_grpc.GreeterStub(grpc.insecure_channel("localhost:" + str(CFG["CAMERA_PORT"])))

    if (CFG["CENT"] == 1 or CFG["CENT"] == 0) and not sent_intrin:  # mqtt intrinsics
        i = stub.get_intrin(pb2.HelloRequest(name="y"))  # seems to work, to test in MOT later
        i_coeffs = i.coeffs.decode("ascii")

        j_i = {
            "width": i.width,
            "height": i.height,
            "ppx": i.ppx,
            "ppy": i.ppy,
            "fx": i.fx,
            "fy": i.fy,
            "coeffs": i_coeffs,
        }
        data_out = json.dumps(j_i)  # encode
        client.publish("i", data_out, retain=True)

        sent_intrin = True

    while True:
        response = stub.get_rframe(pb2.HelloRequest(name="y"))  # in bytes
        frame = imgFromBytes(response.message)

        if CFG["CENT"] == 1 or CFG["CENT"] == 0:  # mqtt and get depth
            response = stub.get_depth(pb2.HelloRequest(name="y"))
            depth = response.img_data.decode("ascii")
            client.publish("i_depth", depth)

        _, frame = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame.tobytes() + b"\r\n")


@app.route("/feed")
def flask_feed():
    # if CFG['DISP'] == 1 or CFG['CENT'] == 1:
    return Response(gather_img_feed(), mimetype="multipart/x-mixed-replace; boundary=frame")
    # else:
    #     gather_img_feed()
    #     return "rip"


sent_intrin = False


def gather_img_feed_depth():
    global sent_intrin
    stub = pb2_grpc.GreeterStub(grpc.insecure_channel("localhost:" + str(CFG["CAMERA_PORT"])))
    dframe = np.zeros((720, 1280), np.uint16)

    if (CFG["CENT"] == 1 or CFG["CENT"] == 0) and not sent_intrin:  # mqtt intrinsics
        i = stub.get_intrin(pb2.HelloRequest(name="y"))  # seems to work, to test in MOT later
        i_coeffs = i.coeffs.decode("ascii")
        j_i = {
            "width": i.width,
            "height": i.height,
            "ppx": i.ppx,
            "ppy": i.ppy,
            "fx": i.fx,
            "fy": i.fy,
            "coeffs": i_coeffs,
        }
        data_out = json.dumps(j_i)  # encode
        client.publish("i", data_out, retain=True)
        sent_intrin = True

    while True:
        response = stub.get_rframe(pb2.HelloRequest(name="y"))  # in bytes
        rframe = imgFromBytes(response.message)

        if CFG["CENT"] == 1 or CFG["CENT"] == 0:  # mqtt and get depth
            response = stub.get_depth(pb2.HelloRequest(name="y"))
            # dframe = imgFromBytes(response.img_data)
            jpg_original = base64.b16decode(response.img_data)
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint16)
            jpg_as_np = np.copy(jpg_as_np).astype("float")
            # dmax = np.max(jpg_as_np) #old code for dynamic max hiding
            jpg_as_np *= 255.0 / CFG["MAX_DEPTH"]
            # print("preclip")
            # print(jpg_as_np)
            jpg_as_np = np.clip(jpg_as_np, 0, 255)

            h, w, _ = rframe.shape
            jpg_as_np = jpg_as_np.reshape(h, w)  # 720,1280)

            # old code for hiding dynamic max in pixels ;failed as it spreads arnd
            # print("d" + str(dmax))
            # tff = math.floor(dmax/255)
            # tff_exceed = 0
            # tff_r = 0
            # if(tff>255):
            #     tff = 255
            #     tff_exceed = tff-255
            # tff_r = dmax - tff*255
            # jpg_as_np[0][0]= tff
            # jpg_as_np[0][1]= tff_exceed
            # jpg_as_np[0][2] = tff_r

            # stacking buffering
            # print("intended")
            # print(jpg_as_np)
            pad = np.zeros((h, w, 2), np.uint16)
            stacked = np.dstack((jpg_as_np, pad))

            # 2d array, vals not accuratec
            # dframe = jpg_as_np.astype('uint8') #.astype('float')
            # dframe = cv2.adaptiveThreshold(dframe, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
            # cv2.imshow("ayto", dframe)

            dframe = stacked

        if dframe is None:
            print("none dframe")

        dframe = dframe.astype("uint8")
        rframe = rframe.astype("uint8")
        frame = cv2.vconcat([rframe, dframe])
        frame = frame.astype("uint8")

        # #testing
        # h,w,_ = frame.shape
        # a =frame[int(h/2):,:]
        # print()
        # print(a.shape)
        # a,_,_ = np.dsplit(a,3)
        # a = a.reshape(int(h/2), w)
        # # a = cv2.cvtColor(jpg_as_np, cv2.COLOR_BGR2GRAY)
        # print("theoretical condfadfsv")
        # print(a.shape)
        # print(a)

        _, frame = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type:image/jpeg\r\n"
            b"Content-Length: " + f"{len(frame)}".encode() + b"\r\n"
            b"\r\n" + frame.tobytes() + b"\r\n"
        )
        # throws weird val
        # yield (b'--frame\r\nContent-T pe: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')


@app.route("/feed_depth")
def flask_feed_depth():
    # if CFG['DISP'] == 1 or CFG['CENT'] == 1:
    return Response(gather_img_feed_depth(), mimetype="multipart/x-mixed-replace; boundary=frame")
    # else:
    #     gather_img_feed_depth()
    #     return "rip"


@app.route("/kill")
def flask_kill():
    client.loop_stop()
    client.disconnect()
    return "yahoo"


if __name__ == "__main__":

    def on_connect(client, userdata, flags, rc):
        print("Connected aaaaaaa")

    with Path("../CONFIG.yml").open() as f:
        CFG = yaml.load(f, Loader=yaml.Loader)
    cmd = "fuser -k " + str(CFG["CLIENT_PORT"]) + "/tcp"
    print(cmd)
    subprocess.run(cmd, shell=True)

    time.sleep(0.1)  # if not the app.run runs first and u die

    client = mqtt.Client()
    client.on_connect = on_connect
    client.connect(str(CFG["MQTT_HOST"]), int(CFG["MQTT_PORT"]))
    client.loop_start()

    logging.basicConfig()
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
