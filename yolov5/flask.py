from flask import Flask, request, jsonify
import cv2
from matplotlib import pyplot as plt
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox
import torch

app = Flask(__name__)

# Replace with the path to your YOLOv5 model weights
weights_path = './runs/train/exp3/weights/best.pt'

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
        _, buffer = cv2.imencode('.jpg', cropped_img)
        img_str = buffer.tobytes()
        result.append(img_str)

    return jsonify({'cropped_images': result})


if __name__ == '__main__':
    app.run(debug=True)
