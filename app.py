from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import torch
import numpy as np
import base64

app = Flask(__name__)
CORS(app)

crop_model_path = '/app/crop.pt'
ocr_model_path = '/app/ocr.pt'

# Load the YOLOv5 models from Ultralytics
crop_model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(crop_model_path))
ocr_model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(ocr_model_path))

def detect_and_crop_objects(image, conf_thres=0.25, iou_thres=0.45, imgsz=640):
    # Perform inference
    results = crop_model(image)
    # Apply non-max suppression to get filtered results
    results = results.xyxy[0].cpu().numpy()  # Get the xyxy format results
    results = results[results[:, 4] > conf_thres]  # Filter by confidence threshold
    cropped_images = []
    # Process detections
    for x1, y1, x2, y2, conf, cls in results:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cropped_img = image[y1:y2, x1:x2]
        cropped_images.append(cropped_img)
    return cropped_images

def perform_ocr_on_image(image):
    # Preprocessing steps
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)
    img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2BGR)
    return "OCR_RESULT_PLACEHOLDER"

@app.route('/detect_and_ocr', methods=['POST'])
def detect_and_ocr():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    image_file = request.files['image']
    image_np = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Detect and crop objects
    cropped_images = detect_and_crop_objects(image)

    result = []
    for cropped_img in cropped_images:
        # Perform OCR on the cropped image
        ocr_result = perform_ocr_on_image(cropped_img)

        # Encode the cropped image in base64 format
        _, buffer = cv2.imencode('.jpg', cropped_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        result.append({
            'cropped_image': img_base64,
            'ocr_result': ocr_result
        })

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
