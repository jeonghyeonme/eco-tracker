from ultralytics import YOLO
import time
import torch
import cv2
import os
from pathlib import Path
import numpy as np

def evaluate_performance(model_path, test_images_dir):
    model = YOLO(model_path)
    
    image_files = list(Path(test_images_dir).glob('*.jpg')) + list(Path(test_images_dir).glob('*.png'))
    
    if not image_files:
        print("No test images found.")
        return

    total_time = 0
    num_images = len(image_files)
    
    print(f"Evaluating model: {model_path}")
    print(f"Testing on {num_images} images...")

    # Warm-up (optional but recommended for accurate timing)
    if torch.cuda.is_available():
        dummy_input = torch.zeros((1, 3, 640, 640)).to('cuda')
        model.model.to('cuda')(dummy_input)

    for img_path in image_files:
        # Measure time
        start_time = time.time()
        
        # YOLOv8 predict includes preprocessing, inference, and postprocessing (NMS)
        results = model.predict(source=str(img_path), imgsz=640, verbose=False)
        
        end_time = time.time()
        total_time += (end_time - start_time)

    avg_inference_time = (total_time / num_images) * 1000 # in ms
    fps = 1.0 / (total_time / num_images)

    print("-" * 30)
    print(f"Performance Metrics for {os.path.basename(model_path)}:")
    print(f"Average Inference Time: {avg_inference_time:.2f} ms")
    print(f"Estimated FPS: {fps:.2f}")
    print("-" * 30)

    # Validate mAP on test set
    print("Running validation on test set...")
    val_results = model.val(data="data/dataset.yaml", split='test')
    print(f"mAP@0.5: {val_results.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {val_results.box.map:.4f}")

if __name__ == "__main__":
    # Example usage:
    # Set model path to your trained weight
    model_weight = "runs/detect/runs/train_yolov8n/weights/best.pt"
    test_dir = "data/test/images"
    
    if os.path.exists(model_weight):
        evaluate_performance(model_weight, test_dir)
    else:
        # Fallback to pretrained if best.pt not found (for debugging)
        print(f"Weight file {model_weight} not found. Checking for alternative...")
        evaluate_performance("yolov8n.pt", test_dir)
