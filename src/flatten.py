import os
import shutil
from tqdm import tqdm  # Progress bar

# Define root paths
root_dir = "C:/Users/mohit/Desktop/Krackhack/Traffic_/data"

# Dataset splits
splits = ["training", "validation", "test"]

# Destination folders
output_dir = os.path.join(root_dir, "datasets")

# Create destination folders
for split in splits:
    os.makedirs(os.path.join(output_dir, f"images/{split}"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, f"labels/{split}"), exist_ok=True)

# Function to move, rename, and flatten files
def collect_files(split):
    frames_dir = os.path.join(root_dir, split, "frames")  # Frames from videos
    annotations_dir = os.path.join(root_dir, split, "annotations")  # Labels

    images_dest = os.path.join(output_dir, f"images/{split}")
    labels_dest = os.path.join(output_dir, f"labels/{split}")

    # Iterate over each subfolder (video)
    for video_folder in tqdm(os.listdir(frames_dir), desc=f"Processing {split} data"):
        video_path = os.path.join(frames_dir, video_folder)
        label_path = os.path.join(annotations_dir, video_folder)

        if not os.path.isdir(video_path) or not video_folder.isdigit():
            continue  # Skip non-directory files and non-numerical folders

        # Process each frame
        for filename in os.listdir(video_path):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                # New file name format: i_originalFilename
                new_filename = f"{video_folder}_{filename}"

                # Copy image
                src_img = os.path.join(video_path, filename)
                dst_img = os.path.join(images_dest, new_filename)
                shutil.copy(src_img, dst_img)

                # Copy and rename corresponding annotation (.txt)
                label_filename = filename.rsplit(".", 1)[0] + ".txt"  # Convert to .txt
                new_label_filename = f"{video_folder}_{label_filename}"  # Match new format
                src_label = os.path.join(label_path, label_filename)
                dst_label = os.path.join(labels_dest, new_label_filename)

                if os.path.exists(src_label):  # Some frames may not have annotations
                    shutil.copy(src_label, dst_label)

# Process all splits
for split in splits:
    collect_files(split)

print("Dataset organized and renamed successfully!")
