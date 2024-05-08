from ultralytics import YOLO
import torch
import cv2
import numpy as np

# Load the YOLO model variant pre-trained on COCO
model = YOLO("yolov8n.pt")

#Apply the model on a custom dataset
#model.train(data = "coco128.yaml", imgsz = (1280,720), epochs = 3, batch_size = 8, weights = "yolov8n.pt", project = "Capra", name = "run1", exist_ok = True)
#metrics = model.val()
results = model(source="C://Users//subbu//OneDrive//Desktop//DTU//MSc.Thesis//Butty//data//data", show=True, conf=0.25, save=True)
path = model.export(format = "onnx")

