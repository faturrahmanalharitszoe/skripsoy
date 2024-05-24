from flask import Flask, request, jsonify
import cv2
from matplotlib import pyplot as plt
import numpy as np
import sys

sys.path.append('C:\\Users\Faturrahman\Downloads\Bahan_Skripsi\yolov5')

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox
import torch

app = Flask(__name__)

# Replace with the path to your YOLOv5 model weights
weights_path = '../yolov5/runs/train/exp3/weights/best.pt'

# Determine device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = attempt_load(weights_path, device=device)
stride = int(model.stride.max())


def detect_and_crop_objects(image_path, conf_thres=0.25, iou_thres=0.45, imgsz=640):
    # Load image
    img0 = cv2.imread(image_path)
    assert img0 is not None, f'Image Not Found {image_path}'

    # Padded resize
    img = letterbox(img0, imgsz, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    # To device
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    cropped_images = []

    # Process detections
    for i, det in enumerate(pred):
        if len(det):
            # Rescale boxes
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = map(int, xyxy)
                cropped_img = img0[y1:y2, x1:x2]
                cropped_images.append(cropped_img)

    return cropped_images


import os

# Define a directory to save cropped images
SAVE_DIR = 'cropped_images'

# Ensure that the directory exists or create it
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

@app.route('/detect_and_crop', methods=['POST'])
def detect_and_crop():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    image_file = request.files['image']
    image_np = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    temp_image_path = 'temp_image.jpg'
    cv2.imwrite(temp_image_path, image)

    cropped_images = detect_and_crop_objects(temp_image_path)

    result = []
    for i, cropped_img in enumerate(cropped_images):
        save_path = os.path.join(SAVE_DIR, f'cropped_{i}.jpg')
        cv2.imwrite(save_path, cropped_img)
        result.append(save_path)

    return jsonify({'cropped_image_paths': result})

from flask import Flask, request, jsonify
import cv2
import torch
import pandas as pd
import numpy as np
from models.experimental import attempt_load

app = Flask(__name__)

# Correctly set paths for compatibility on non-Unix systems
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='../yolov5/runs/train/exp37/weights/best.pt')

@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    
    # Load the image from the request
    image_file = request.files['image']
    image_np = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Preprocessing steps
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_clahe = clahe.apply(img_gray)
    img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2BGR)

    # Perform inference
    results = model(img_clahe)

    # Filter detections with confidence > 0.6
    filtered_results = results.xyxy[0][results.xyxy[0][:, 4] > 0.6]

    # Sort the filtered results horizontally by the x-axis of the bounding box
    sorted_filtered_results = filtered_results[filtered_results[:, 0].argsort()]

    # Extracting details for display
    detections = sorted_filtered_results.cpu().numpy()

    # Extract class names in the order of sorted detections by X1
    sorted_class_names = [model.names[int(cls)] for cls in sorted_filtered_results[:, -1]]

    # Combine class names into a single string, preserving their left-to-right order in the image
    combined_classes_text = ''.join(sorted_class_names)

    # Prepare the JSON response
    response = {
        'detections': detections.tolist(),
        'combined_classes_text': combined_classes_text
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

