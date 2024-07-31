import cv2
import grpc
import newMOT_pb2 as motor_pb2
import newMOT_pb2_grpc as motor_pb2_grpc
import base64

def main():
    # Connect to the gRPC server
    channel = grpc.insecure_channel('localhost:50052')
    stub = motor_pb2_grpc.MOTorStub(channel)

    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert the frame to bytes
        _, buffer = cv2.imencode('.jpg', frame)
        data = base64.b64encode(buffer)

        # Create the request
        request = motor_pb2.MOT_pic(image=data)

        # Send the request and get the response
        response = stub.askInfer_MOT(request)

        # Process the response (e.g., draw bounding boxes)
        for detection in response.detections:
            x_low, y_low = int(detection.x_low), int(detection.y_low)
            x_high, y_high = int(detection.x_high), int(detection.y_high)
            cv2.rectangle(frame, (x_low, y_low), (x_high, y_high), (0, 255, 0), 2)

        # Display the frame with bounding boxes
        cv2.imshow('Webcam Stream', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()