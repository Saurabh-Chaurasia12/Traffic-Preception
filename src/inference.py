import torch
from ultralytics import YOLO
import cv2
from pathlib import Path

def load_model(weights_path):
    model = YOLO(weights_path)
    model.eval()
    return model

def predict(model, image_path, device):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(image_tensor)

    return prediction

def main():
    weights_path = 'weights/yolov8n.pt'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = load_model(weights_path).to(device)

    test_image_path = Path('data/augmented_frames/frame_0001.jpg')
    prediction = predict(model, test_image_path, device)
    print(prediction)

if __name__ == "__main__":
    main()
