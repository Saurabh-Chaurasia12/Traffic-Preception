import torch
import os
from ultralytics import YOLO

# Ensure weights directory exists
WEIGHTS_DIR = "C:/Users/mohit/Desktop/Krackhack/Traffic_/weights/"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Define dataset path
DATA_YAML = "C:/Users/mohit/Desktop/Krackhack/Traffic_/data/datasets/custom.yaml"

# Load YOLOv8 model (Choose from yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
model = YOLO("C:/Users/mohit/Desktop/Krackhack/Traffic_/models/yolov8n.pt")  # Using nano model for faster training

print("✅ Model Loaded Successfully!")

# Train the model
print("🚀 Training Started...")
model.train(
    data=DATA_YAML,
    epochs=100,
    batch=16,
    imgsz=640,
    device="cuda" if torch.cuda.is_available() else "cpu",
    project="runs/detect",
    name="train_custom",
    exist_ok=True  # Overwrites previous runs if needed
)

print("🎯 Training Completed Successfully!")

# Evaluate the model
print("📊 Running Model Evaluation...")
metrics = model.val()
print(f"📈 Evaluation Metrics: {metrics}")

# Save the trained model inside weights/ folder
MODEL_PATH = os.path.join(WEIGHTS_DIR, "yolov8_custom_trained.pt")
model.save(MODEL_PATH)

print(f"✅ Trained Model Saved at: {MODEL_PATH}")

# Export the trained model (optional)
EXPORT_PATH = os.path.join(WEIGHTS_DIR, "yolov8_custom_trained.torchscript")
model.export(format="torchscript", dynamic=True, simplify=True, optimize=True, out=EXPORT_PATH)

print(f"✅ Model Exported as TorchScript at: {EXPORT_PATH}")
print("🚀 Training Pipeline Completed Successfully! 🎉")
