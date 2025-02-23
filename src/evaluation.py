import torch
from ultralytics import YOLO

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained YOLOv8 model
    model = YOLO("weights/yolov8_custom_trained.pt")  # Path to your trained model

    print("✅ Model Loaded Successfully!")

    # Run evaluation on the test dataset
    print("📊 Running Evaluation on Test Data...")
    metrics = model.val(data="data/datasets/custom.yaml", split="test", device=device)

    # Extract important evaluation metrics
    precision = metrics.box.map50  # Precision at IoU 0.5
    recall = metrics.box.map  # Recall
    map_50 = metrics.box.map50  # mAP at IoU 0.5
    map_50_95 = metrics.box.map  # mAP at IoU 0.5:0.95

    print("\n📊 Evaluation Results:")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall: {recall:.4f}")
    print(f"✅ mAP@50: {map_50:.4f}")
    print(f"✅ mAP@50-95: {map_50_95:.4f}")

if __name__ == "__main__":
    evaluate_model()
