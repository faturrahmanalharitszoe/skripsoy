import torch
from pathlib import Path
import numpy as np
import cv2
# YOLOv5 imports might need to adjust based on your directory structure
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load('./runs/train/exp3/weights/best.pt', device=device)
stride = int(model.stride.max())

def detect_and_crop_objects(image_path, conf_thres=0.25, iou_thres=0.45, imgsz=640):
    img0 = cv2.imread(image_path)
    assert img0 is not None, f'Image Not Found {image_path}'
    img = letterbox(img0, imgsz, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    cropped_images = []
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = map(int, xyxy)
                cropped_img = img0[y1:y2, x1:x2]
                cropped_images.append(cropped_img)
    return cropped_images
