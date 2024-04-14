import cv2
import torch
import pandas as pd

# Assuming the model is already loaded as in the previous snippet
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./runs/train/exp37/weights/best.pt')

def enhanced_object_detection(image_path):
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)
    img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2BGR)
    
    results = model(img_clahe)
    detections = results.xyxy[0][results.xyxy[0][:, 4] > 0.6].cpu().numpy()
    
    # Sort the detections by X1 (the horizontal position)
    sorted_detections = detections[detections[:, 0].argsort()]
    
    # Extract class names based on detections, sorted by X1
    sorted_class_names = [model.names[int(cls)] for cls in sorted_detections[:, -1]]
    
    # Combine class names into a single string, preserving their order
    combined_classes_text = ' '.join(sorted_class_names)
    
    # For returning the text, you might also want to return any other relevant information
    return {'detected_classes_sorted_by_x1': combined_classes_text}
