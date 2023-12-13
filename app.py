from flask import Flask, render_template, Response, send_from_directory
import cv2
import numpy as np
import tensorflow as tf
import reid_model
import time
import torch
import glob
import os
import socket
import struct
import pickle

app = Flask(__name__)


def find_closest_tensor(feature_tensor, feature_bank_path):
    os.makedirs(feature_bank_path, exist_ok=True)
    feature_bank =  glob.glob(os.path.join(feature_bank_path, "*.pth"))
    
    min_distance = float('inf')
    
    for tensor_path in feature_bank:
        compare_tensor = torch.load(tensor_path)
        distance = reid_model.euclidean_distance(feature_tensor, compare_tensor)
        
        if distance < min_distance:
            min_distance = distance
            
    return  min_distance
        
def is_new_person(feature_tensor, feature_bank_path, threshold=1.06):
    min_distance = find_closest_tensor(feature_tensor, feature_bank_path)
    
    if min_distance < threshold:
        return False
    
    return True

def save_feature_tensor(feature_tensor, feature_bank_path):
    #count number of files => new id
    os.makedirs(feature_bank_path, exist_ok=True)
    feature_bank = glob.glob(os.path.join(feature_bank_path, "*.pth"))
    new_id = len(feature_bank) + 1

    filename = os.path.join(feature_bank_path, f'id_{str(new_id)}_{time.time()}_feature_tensor.pth') # id_timestamp_feature_tensor.pth
    torch.save(feature_tensor, filename)

def save_image(cropped_frame, id, cam_id):
    #count number of files => new id
    os.makedirs('./images/', exist_ok=True)

    filename = os.path.join('./images/', f'id_{str(id-1)}_unnamed_{cam_id}_{time.time()}_image.jpg') # id_timestamp_feature_tensor.pth
    cv2.imwrite(filename, cropped_frame)

def get_new_id(feature_bank_path):
    os.makedirs(feature_bank_path, exist_ok=True)
    feature_bank = glob.glob(os.path.join(feature_bank_path, "*.pth"))

    return len(feature_bank) + 1
        
def get_pid(feature_tensor, feature_bank_path):
    os.makedirs(feature_bank_path, exist_ok=True)
    feature_bank =  glob.glob(os.path.join(feature_bank_path, "*.pth"))

    distances = []

    for tensor in feature_bank:
        compare_tensor = torch.load(tensor)
        distance = reid_model.euclidean_distance(feature_tensor, compare_tensor)
        distances.append(distance)
        
    index = distances.index(min(distances))
    return feature_bank[index].split("_")[1]

def connect_socket(cam_ip, port):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((cam_ip, port))
    return client_socket



camera_sources = [
    {'ip': '192.168.110.108', 'port': 9999}
    # {'ip': '192.168.0.175', 'port': 9999}
    # {'ip': '192.168.243.98', 'port': 9999},
    # {'ip': '192.168.243.98', 'port': 9999}
]

client_sockets = []
for socket_index in range(1):
    cam_ip = camera_sources[socket_index]['ip']
    port = camera_sources[socket_index]['port']
    client_sockets.append(connect_socket(cam_ip, port))

feature_bank_path = './bank/'


def generate_frames(cam_index, cam_ip, port):
    # Set up the client socket
    data = b""
    payload_size = struct.calcsize("L")

    while True:
        try:
            # Receive the size of the frame data
            while len(data) < payload_size:
                data += client_sockets[cam_index-1].recv(4096)

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]

            # Unpack the frame and tensor sizes
            msg_size = struct.unpack("L", packed_msg_size)[0]

            # Receive the frame and tensor data
            while len(data) < msg_size:
                data += client_sockets[cam_index-1].recv(4096)

            
            socket_data = data[:msg_size]
            data = data[msg_size:]

            # Deserialize the frame and tensor data
            socket_data = pickle.loads(socket_data)

            # Extract socket data
            frame = pickle.loads(socket_data['frame'])
            bounding_boxes = pickle.loads(socket_data['bounding_boxes'])
            feature_tensors = pickle.loads(socket_data['feature_tensors'])
            images = pickle.loads(socket_data['images'])

            for box, feature_tensor, image in zip(bounding_boxes, feature_tensors, images):
                left, top, right, bottom = pickle.loads(box)

                feature_tensor = pickle.loads(feature_tensor)

                id = "unidentified"
                #get id 
                if is_new_person(feature_tensor, feature_bank_path):
                    save_feature_tensor(feature_tensor, feature_bank_path)
                    print("Save new person tensor")
                    id = get_new_id(feature_bank_path)
                    saved_image = pickle.loads(image)
                    save_image(saved_image, id, cam_index)
                    print("Save new person image")
                else:
                    id = get_pid(feature_tensor, feature_bank_path)

                #draw
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {id}', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        except pickle.UnpicklingError as e:
            print(f"Error unpickling frame: {e}")
            #handling error
            continue

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed/<int:cam_index>')
def video_feed(cam_index):
    camera_source = camera_sources[cam_index - 1]
    cam_ip = camera_source['ip']
    port = camera_source['port']
    return Response(generate_frames(cam_index, cam_ip, port), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory('images', filename)

@app.route('/images')
def show_images():
    image_folder = './images/'
    image_names = os.listdir(image_folder)
    return render_template('images.html', image_names=image_names)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False, port=5000)
