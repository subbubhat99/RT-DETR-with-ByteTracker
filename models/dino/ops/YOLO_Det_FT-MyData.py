from ultralytics import YOLO
import torch
import cv2
from ultralytics.models.yolo import model
import numpy as np

# Load the YOLO model variant pre-trained on COCO
model = YOLO("yolov8n.pt")

#Apply the model on a custom dataset
train_results = model.train(data = "C://Users//subbu//OneDrive//Desktop//DTU//MSc.Thesis//capra.yaml", imgsz = (640,480), epochs = 3, name = "run1")
val_results = model.val()
#path = model.export(format = "onnx")
print(val_results)


