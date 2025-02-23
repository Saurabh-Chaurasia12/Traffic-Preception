import cv2
import json
from pathlib import Path
import albumentations as A
import shutil

def extract_frames(video_path, output_folder, interval=30):
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frame_path = output_folder / f"frame_{frame_count:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
        frame_count += 1
    cap.release()

def load_gps_data(json_path):
    with open(json_path, 'r') as file:
        return json.load(file)

def augment_data(image_folder, output_folder, augmentations):
    for image_path in image_folder.glob('*.jpg'):
        image = cv2.imread(str(image_path))
        augmented = augmentations(image=image)
        augmented_image = augmented['image']
        augmented_path = output_folder / f"{image_path.stem}_augmented.jpg"
        cv2.imwrite(str(augmented_path), augmented_image)

def process_split(split_folder, augmentations_train, augmentations_val_test):
    camera_videos = split_folder / 'camera_videos'
    frames_folder = split_folder / 'frames'
    augmented_folder = split_folder / 'augmented_frames'

    frames_folder.mkdir(parents=True, exist_ok=True)
    augmented_folder.mkdir(parents=True, exist_ok=True)

    for video_path in camera_videos.glob('*.mp4'):
        video_name = video_path.stem
        frames_output = frames_folder / video_name
        augmented_output = augmented_folder / video_name

        frames_output.mkdir(parents=True, exist_ok=True)
        augmented_output.mkdir(parents=True, exist_ok=True)

        extract_frames(video_path, frames_output)

        if split_folder.name == "train":
            augment_data(frames_output, augmented_output, augmentations_train)
        else:
            # No augmentations for val and test
            for image_path in frames_output.glob('*.jpg'):
                shutil.copy(image_path, augmented_output / image_path.name)

def main():
    data_folder = Path('data')
    train_folder = data_folder / 'training'
    val_folder = data_folder / 'validation'
    test_folder = data_folder / 'test'

    augmentations_train = A.Compose([
        A.RandomCrop(width=450, height=450),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])

    augmentations_val_test = A.Compose([])

    process_split(train_folder, augmentations_train, augmentations_val_test)
    process_split(val_folder, augmentations_train, augmentations_val_test)
    process_split(test_folder, augmentations_train, augmentations_val_test)

    #Load gps data from the train folder.
    # gps_data = load_gps_data(train_folder / 'gps_json' / 'example.json')

if __name__ == "__main__":
    main()