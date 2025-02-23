import cv2
from pathlib import Path
from ultralytics import YOLO

def generate_annotations(image_folder, output_folder, model_path='models/yolov8n.pt', conf_threshold=0.25, class_names=None):
    """Generates YOLO format annotations using a pre-trained model."""
    model = YOLO(model_path)
    
    # If class_names is not provided, use model's default class names
    if class_names is None:
        class_names = model.names
    
    for image_path in image_folder.glob('*.jpg'):
        image = cv2.imread(str(image_path))
        results = model(image, conf=conf_threshold)
        print(f"Predictions for {image_path}: {len(results[0].boxes)}")
        
        annotated_image = image.copy()
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_id = int(box.cls[0].item())
                
                # Get class name from model's names dictionary
                class_label = model.names[class_id] if class_id in model.names else str(class_id)
                
                # Draw rectangle and label
                cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(annotated_image, class_label, (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imwrite(str(output_folder / f"{image_path.stem}_annotated.jpg"), annotated_image)
        
        # Write YOLO format annotations
        annotation_path = output_folder / f"{image_path.stem}.txt"
        with open(annotation_path, 'w') as f:
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    class_id = int(box.cls[0].item())
                    # Convert to YOLO format (normalized coordinates)
                    image_height, image_width, _ = image.shape
                    x_center = ((x1 + x2) / 2) / image_width
                    y_center = ((y1 + y2) / 2) / image_height
                    width = (x2 - x1) / image_width
                    height = (y2 - y1) / image_height
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def process_split(split_folder, model_path, class_names=None):
    """Processes a single split (train, val, test)."""
    frames_folder = split_folder / 'frames'
    output_folder = split_folder / 'annotations'
    output_folder.mkdir(parents=True, exist_ok=True)
    
    for video_folder in frames_folder.iterdir():
        if video_folder.is_dir():
            video_output_folder = output_folder / video_folder.name
            video_output_folder.mkdir(parents=True, exist_ok=True)
            generate_annotations(video_folder, video_output_folder, model_path, class_names=class_names)

def main():
    data_folder = Path('data')
    train_folder = data_folder / 'training'
    val_folder = data_folder / 'validation'
    test_folder = data_folder / 'test'
    model_path = 'models/yolov8n.pt'
    
    # Load the model once to get its class names
    model = YOLO(model_path)
    class_names = model.names  # This will get the default COCO class names
    
    # Process each split using the model's class names
    process_split(train_folder, model_path, class_names)
    process_split(val_folder, model_path, class_names)
    process_split(test_folder, model_path, class_names)

if __name__ == "__main__":
    main()