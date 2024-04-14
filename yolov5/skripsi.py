import torch
import numpy as np
import cv2
import easyocr
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox
import argparse

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def load_model(weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = attempt_load(weights_path, device=device)
    stride = int(model.stride.max())
    return model, stride, device

def detect_and_crop_objects(model, stride, device, image_path, conf_thres=0.25, iou_thres=0.45, imgsz=640):
    img0 = cv2.imread(image_path)
    assert img0 is not None, f'Image Not Found {image_path}'

    img = letterbox(img0, imgsz, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # Normalize
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

def apply_mask_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized_gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    _, img_plate_bw = cv2.threshold(normalized_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_plate_bw = cv2.morphologyEx(img_plate_bw, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(img_plate_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100: continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if 0.2 < aspect_ratio < 1.0 and gray.shape[0] * 0.3 < h < gray.shape[0] * 0.8 and gray.shape[1] * 0.02 < w < gray.shape[1] * 0.15:
            filtered_contours.append(contour)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, filtered_contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    img_masked = cv2.bitwise_and(image, image, mask=mask)
    return img_masked

def extract_text_from_image_with_masking(cropped_images):
    reader = easyocr.Reader(['en'], gpu=True if torch.cuda.is_available() else False)
    combined_text = ""
    for img in cropped_images:
        masked_img = apply_mask_for_ocr(img)
        results = reader.readtext(masked_img)
        for (_, text, _) in results:
            combined_text += text + " "
    return combined_text.strip()

def process_image_with_masking(image_path, weights_path):
    model, stride, device = load_model(weights_path)
    cropped_images = detect_and_crop_objects(model, stride, device, image_path)
    detected_text = extract_text_from_image_with_masking(cropped_images)
    return detected_text

def parse_args():
    parser = argparse.ArgumentParser(description="Detect and extract text from images.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to the YOLO model weights")
    return parser.parse_args()

def main():
    args = parse_args()
    detected_text = process_image_with_masking(args.image_path, args.weights_path)
    print(f"Detected Text: {detected_text}")

if __name__ == "__main__":
    main()
