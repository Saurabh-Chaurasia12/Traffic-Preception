import cv2
import torch
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO("weights/yolov8_custom_trained.pt")  # Path to trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the input video path
video_path = "C:/Users/mohit/Desktop/Krackhack/Traffic_/data/test/camera_videos/cars.mp4"  # Change this to your video file

# Open the video file
cap = cv2.VideoCapture(video_path)

# Class names from model
class_names = model.names  # Ensure this matches your custom dataset

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if video ends

    # Perform YOLOv8 detection
    results = model(frame)

    # Draw bounding boxes and labels
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes (x1, y1, x2, y2)
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs

        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = f"{class_names[class_id]}: {score:.2f}"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show real-time frame with detections
    cv2.imshow("YOLOv8 Real-Time Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print("✅ Real-Time Detection Completed!")
