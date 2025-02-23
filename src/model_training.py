import torch
import os
import glob
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
    epochs=10,
    batch=16,
    workers=0,
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



# Define paths correctly
MODELS_DIR = "C:/Users/mohit/Desktop/Krackhack/Traffic_/models"
custom_export_path = os.path.join(WEIGHTS_DIR, "yolov8_custom_trained.torchscript")

# Export the trained model
print("📦 Exporting Model...")
exported_files = model.export(format="torchscript", simplify=True, optimize=True)

# Capture exported file path
if isinstance(exported_files, list) and exported_files:
    exported_model_path = exported_files[0]
else:
    exported_model_path = os.path.join(MODELS_DIR, "yolov8n.torchscript")  # Default YOLO export name

# Check if the exported file exists and rename it
if os.path.exists(exported_model_path):
    os.rename(exported_model_path, custom_export_path)
    print(f"✅ Model Exported as TorchScript at: {custom_export_path}")
else:
    print("⚠️ Exported file not found. Checking models/ directory...")
    exported_candidates = glob.glob(os.path.join(MODELS_DIR, "*.torchscript"))  # Look in models/
    
    if exported_candidates:
        os.rename(exported_candidates[0], custom_export_path)
        print(f"✅ Found and renamed exported file: {custom_export_path}")
    else:
        print("❌ No exported files found. Please check the export process.")

print("🚀 Training Pipeline Completed Successfully! 🎉")

