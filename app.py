from flask import Flask, render_template, Response, request, jsonify
import torch
from ultralytics import YOLO
import cv2
import os
from pathlib import Path
from werkzeug.utils import secure_filename
import threading

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Load YOLOv8 model
model = YOLO('weights/yolov8_custom_trained.pt')  # Update with your trained model path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Global variables
video_path = None
streaming = False
object_counts = {}  # Stores count of each object type

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Handle video uploads."""
    global video_path, streaming
    streaming = False  # Reset streaming flag

    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)

    print(f"✅ Video uploaded: {video_path}")
    return jsonify({"message": "File uploaded successfully"}), 200

@app.route('/start_stream', methods=['POST'])
def start_stream():
    """Start video streaming when button is clicked."""
    global streaming
    if video_path is None:
        return jsonify({"error": "No video uploaded"}), 400

    streaming = True  # Enable streaming
    return jsonify({"message": "Streaming started"}), 200

def generate_frames():
    """Process video and stream frames with bounding boxes and object counts."""
    global video_path, streaming, object_counts
    if video_path is None:
        print("❌ No video path set!")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Cannot open video file!")
        return

    print(f"✅ Streaming video: {video_path}")

    while cap.isOpened():
        if not streaming:
            break  # Stop streaming if flag is False

        ret, frame = cap.read()
        if not ret:
            print("✅ Video streaming completed!")
            break

        # Run YOLOv8 on the frame
        results = model(frame)

        # Reset counts before processing a new frame
        object_counts.clear()

        # Process detections
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
            class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores

            for box, class_id, confidence in zip(boxes, class_ids, confidences):
                x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
                label = model.names[class_id]  # Get class label
                confidence_text = f"{confidence:.2f}"  # Format confidence

                # Update object counts
                object_counts[label] = object_counts.get(label, 0) + 1

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Display label and confidence
                text = f"{label}: {confidence_text}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Overlay counts on the frame
        y_offset = 30
        for label, count in object_counts.items():
            cv2.putText(frame, f"{label}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            y_offset += 30

        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield frame in the required format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_counts', methods=['GET'])
def get_counts():
    """Return the current object counts."""
    return jsonify(object_counts)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run()
