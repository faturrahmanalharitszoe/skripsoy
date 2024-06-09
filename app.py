from flask import Flask, request, jsonify
import cv2
import torch
import numpy as np
import os
import pathlib

app = Flask(__name__)

crop_model_path = '/crop.pt'
ocr_model_path = '/ocr.pt'

# Load the YOLOv5 model from Ultralytics
crop_model = torch.hub.load('ultralytics/yolov5', 'custom', path=crop_model_path)
ocr_model = torch.hub.load('ultralytics/yolov5', 'custom', path=ocr_model_path)

# def detect_and_crop_objects(image, conf_thres=0.25, iou_thres=0.45, imgsz=640):
#     # Perform inference
#     results = crop_model(image)
#     # Apply non-max suppression to get filtered results
#     results = results.xyxy[0].cpu().numpy()  # Get the xyxy format results
#     results = results[results[:, 4] > conf_thres]  # Filter by confidence threshold
#     cropped_images = []
#     # Process detections
#     for x1, y1, x2, y2, conf, cls in results:
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#         cropped_img = image[y1:y2, x1:x2]
#         cropped_images.append(cropped_img)
#     return cropped_images

# # Define a directory to save cropped images
# SAVE_DIR = 'cropped_images'

# # Ensure that the directory exists or create it
# if not os.path.exists(SAVE_DIR):
#     os.makedirs(SAVE_DIR)

# @app.route('/detect_and_crop', methods=['POST'])
# def detect_and_crop():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image uploaded'})
#     image_file = request.files['image']
#     image_np = np.frombuffer(image_file.read(), np.uint8)
#     image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
#     cropped_images = detect_and_crop_objects(image)

#     result = []
#     for i, cropped_img in enumerate(cropped_images):
#         save_path = os.path.join(SAVE_DIR, f'cropped_{i}.jpg')
#         cv2.imwrite(save_path, cropped_img)
#         result.append(save_path)

#     return jsonify({'cropped_image_paths': result})

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
    results = ocr_model(img_clahe)

    # Filter detections with confidence > 0.6
    filtered_results = results.xyxy[0][results.xyxy[0][:, 4] > 0.6]

    # Sort the filtered results horizontally by the x-axis of the bounding box
    sorted_filtered_results = filtered_results[filtered_results[:, 0].argsort()]

    # Extracting details for display
    detections = sorted_filtered_results.cpu().numpy()

    # Extract class names in the order of sorted detections by X1
    sorted_class_names = [ocr_model.names[int(cls)] for cls in sorted_filtered_results[:, -1]]

    # Combine class names into a single string, preserving their left-to-right order in the image
    combined_classes_text = ''.join(sorted_class_names)

    # Prepare the JSON response
    response = {
        'detections': detections.tolist(),
        'combined_classes_text': combined_classes_text
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("5000"),debug=True)
