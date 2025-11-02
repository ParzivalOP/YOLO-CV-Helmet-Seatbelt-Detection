from roboflow import Roboflow
from ultralytics import YOLO
import os
rf = Roboflow(api_key="roboflow api key")
from ultralytics import YOLO

model = YOLO("yolov8s.pt") 

model.train(
    data=f"{dataset_path}/data.yaml",
    epochs=12,
    imgsz=640,
    batch=8,
    project="Helmet_Seatbelt",
    name="helmet_seatbelt_v1",
    split=0.2
)

import shutil
import os


trained_model_path = "/helmet_seatbelt_v12/weights/best.pt"

# Destination folder in Google Drive
destination_folder = "MyDrive/YOLOv8trained model"
os.makedirs(destination_folder, exist_ok=True)

# Copy the model
shutil.copy(trained_model_path, os.path.join(destination_folder, "best.pt"))

print("Fine-tuned model saved to Google Drive at:", destination_folder)
