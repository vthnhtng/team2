import cv2
import socket
import pickle
import struct
import numpy as np
from ultralytics import YOLO
import torch
import reid_model
from reid_model import PersonReidModel
import time

def crop_bounding_box(frame, left, top, right, bottom, normalized_size=(128, 256)):
    # Crop the bounding box from the frame
    cropped_frame = frame[top:bottom, left:right]

    # Resize the cropped region to the specified size
    resized_cropped_frame = cv2.resize(cropped_frame, normalized_size)

    return resized_cropped_frame  # ndarray



# Load the YOLOv8 model
yolo_model = YOLO('yolov8x.pt')

# reid model
reid_pretrained_path = "./pretrained_models/resnet50-19c8e357.pth"
reid_trained_path = "./trained_models/market_resnet50_model_120_rank1_945.pth"
reid = PersonReidModel(reid_pretrained_path, reid_trained_path)


def start_server():
    # Set up the server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 9999))
    server_socket.listen(0)

    # Set up the webcam
    cap = cv2.VideoCapture(0)

    while True:
        try:
            print("Listening")
            # Accept a client connection
            connection, addr = server_socket.accept()
            print(f"Connection from {addr}")

            while True:
                # Read a frame from the webcam
                success, frame = cap.read()
                if not success:
                    break
                
                detections = yolo_model.track(frame, persist=True, tracker='bortsort.yaml', verbose=False)
                # Extract bounding boxes, classes, names, and confidences
                boxes = detections[0].boxes.xyxy
                
                feature_tensors = []
                bounding_boxes = []
                images = []
                # Iterate through detected objects and draw bounding boxes
                for i in range(len(boxes)):
                    clss = detections[0].boxes.cls[i]
                    conf = detections[0].boxes.conf[i]
                    left, top, right, bottom = boxes[i]
                    left, top, right, bottom = int(left), int(top), int(right), int(bottom)
                    if int(clss) == 0 and float(conf) > 0.7:
                    # prepare input
                        bounding_box = crop_bounding_box(frame, left, top, right, bottom)  # return ndarray
                        bounding_box_tensor = reid_model.tensor_from_ndarray(bounding_box)  # transform data
                        feature_tensor = reid.perform_inference(bounding_box_tensor)

                        bounding_boxes.append(np.array([left, top, right, bottom]))
                        feature_tensors.append(feature_tensor)
                        images.append(bounding_box)

                # sending feature tensor from inference of model
                for i in range(len(bounding_boxes)):
                    bounding_boxes[i] = pickle.dumps(bounding_boxes[i])

                for i in range(len(feature_tensors)):
                    feature_tensors[i] = pickle.dumps(feature_tensors[i])
                    
                for i in range(len(images)):
                    images[i] = pickle.dumps(images[i])

                # sending_frame
                data = {
                    'frame': pickle.dumps(frame),
                    'bounding_boxes': pickle.dumps(bounding_boxes),
                    'feature_tensors': pickle.dumps(feature_tensors),
                    'images': pickle.dumps(images)
                }

                serialized_data = pickle.dumps(data)

                # Pack the length of the serialized data
                message_size = struct.pack("L", len(serialized_data))

                # Send the message size and serialized data to the client
                connection.sendall(message_size + serialized_data)

        except (socket.error, ConnectionResetError):
            # Handle client disconnection
            print("Client disconnected")

        finally:
            connection.close()

            # Release the resources
    cap.release()


if __name__ == "__main__":
    start_server()
