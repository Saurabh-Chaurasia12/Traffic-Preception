# Intelligent Traffic Perception
![cover_image_raw](https://github.com/user-attachments/assets/4cf37c66-a437-454d-99e0-cbad25f3fb4c)

## 🔍 Overview
With increasing urban congestion, real-time traffic monitoring is crucial for improving road safety and efficiency. This project leverages YOLOv8 on the BDDA dataset to develop an advanced object detection system capable of identifying multiple traffic elements, including vehicles, pedestrians, traffic signs, lanes, and object speeds. The system is designed to assist law enforcement, city planners, and autonomous vehicles by providing structured insights from live traffic video feeds.


## 🎯 Objectives
* **YOLOv8 Model Selection and Assessment:** Commencing with the selection of a pre-trained YOLOv8n model and evaluating its baseline performance on the BDD100K dataset for object detection purposes.
* **Prepare BDDA Dataset For Model:** Assembling and annotating a targeted dataset dedicated to vehicles to enhance the model's detection accuracy for a range of vehicle types.
* **Model Refinement for Superior Detection:** Applying transfer learning techniques to fine-tune the YOLOv8n model, with a special focus on detecting vehicles from aerial views, thus significantly improving precision and recall rates.
* **Thorough Evaluation of Model Performance:** Conducting a detailed analysis of learning curves, confusion matrices, and performance metrics to ensure the model's reliability and its capability to generalize.
* **Generalization and Inference on Test Data:** Verifying the model's robustness through generalization tests using validation images, an unseen test image, and a test video, showcasing its real-world applicability and efficiency.
* **Real-Time Traffic Analysis:** Developing an algorithm to quantify traffic density by real-time vehicle counting and traffic intensity analysis on test video footage.
* **Preparation for Cross-Platform Deployment:** Exporting the optimized model in the ONNX format to ensure cross-platform compatibility and facilitate deployment across diverse environments.


## 📚 Dataset Description

### 🌐 Overview
**BDDA (Berkeley DeepDrive A)** is one of the largest open-source datasets. It consists of 2,000 driving videos collected from real-world driving scenarios across diverse geographic locations, weather conditions, and times of day. The dataset is designed to support a variety of autonomous driving tasks, such as object detection, lane detection, instance segmentation, and trajectory prediction.

BDDA provides high-quality annotations for multiple tasks, making it an invaluable resource for training and evaluating deep learning models in autonomous driving.

### 🔍 Specifications 
- 🚗 **Class**: Vehicles (cars, trucks, buses, pedestrians, cyclists, etc.)
- 🖼️ **Total Video**: 1800 Video
- 📏 **Image Dimensions**: 1280 × 720 pixels
- 📂 **Format**: YOLOv8 annotation format

### 🔄 Pre-processing
To preprocess the BDD100K video dataset and convert videos into individual image frames with a frame interval of 30,
---------------------------
-------------------
---------------

### 🔢 Dataset Split
The dataset is already split into:
- **Training Set**: 950++ videos for model training with diverse scenarios.
- **Validation Set**: 250++ videos for cross validation unbiased model performance evaluation.
- **Test Set**: 450++ videos for unbiased model performance evaluation.

### 🚀 Significance
This dataset is pivotal in developing sophisticated vehicle detection models and shaping intelligent transportation systems for smarter city infrastructures.

### 📌 Access
The dataset is publicly available berkeley.edu website:
- BDDA Dataset: [Dashboard Cam Vehicle Detection Video Dataset](https://bdd-data.berkeley.edu/)

# BADAL
## 📁 File Descriptions

- **`images/`**: This directory houses the cover images for the project and the sample image utilized within the notebook.
- **`models/`**: Contains the best-performing fine-tuned YOLOv8 model in both `.pt` (PyTorch format) and `.onnx` (Open Neural Network Exchange format) for broad compatibility.
- **`LICENSE`**: The legal framework defining the terms under which this project's code and dataset can be used.
- **`README.md`**: The document you are reading that offers an insightful overview and essential information about the project.
- **`real_time_traffic_analysis.py`**: The Python script for deploying the YOLOv8 model to estimate traffic density in real-time on a local system.
- **`real-time_traffic_density_estimation_yolov8.ipynb`**: The Jupyter notebook that documents the model development pipeline, from data preparation to model evaluation and inference.
- **`Running_Real-Time_Traffic_Analysis.gif`**: A GIF demonstration showing the real-time traffic analysis capability of our model when the `real_time_traffic_analysis.py` script is executed.
- **`sample_video.mp4`**: The video file used for testing the traffic estimation algorithm and the deployment code in `real_time_traffic_analysis.py`.

# BADAL

## 🚀 Instructions for Local Execution

To experience the full capabilities of the YOLOv8 Traffic Density Estimation project on your local machine, follow these steps:

### 1️⃣. Initial Setup
1. **Clone the Repository**: Start by cloning the project repository to your local system using the command below:
    ```bash
    git clone https://github.com/Saurabh-Chaurasia12/Traffic-Preception.git
    ```
2. **Navigate to the Project Directory**: After cloning, change into the project directory with:
    ```bash
    cd Traffic-Preception
    ```
# BADAL

### 2️⃣. Exploring the Model Development Pipeline
Get hands-on with the model development process and see the results of traffic density estimation:
1. **Download the Dataset**: Access the dataset from [Kaggle](https://www.kaggle.com/datasets/farzadnekouei/top-view-vehicle-detection-image-dataset). Download and extract it to a known directory on your machine.
2. **Open the Notebook**: Launch Jupyter Notebook or JupyterLab and open `real-time_traffic_density_estimation_yolov8.ipynb` to explore the model development pipeline.
3. **Install Dependencies**: Ensure all necessary Python libraries are installed for flawless execution.
4. **Update Paths**: Update the paths in the notebook for the dataset, sample image, and sample video to their respective locations on your local system.
5. **Run the Notebook**: Execute all cells in the notebook to step through the data preprocessing, model training, and evaluation phases.

### 3️⃣. Watching the Real-Time Performance
Witness the real-time traffic analysis capability of our YOLOv8 model:
1. **Install Ultralytics YOLO**: Ensure you have the `ultralytics` package installed by running:
    ```bash
    pip install ultralytics
    ```
2. **Run the Analysis Script**: Execute the script to start the real-time traffic density estimation:
    ```bash
    python real_time_traffic_analysis.py
    ```
3. **Real-Time Analysis**: The video window will display the live traffic analysis. To exit, simply press 'q' while the video window is active.

This GIF showcases our algorithm running in real-time:

![Real-Time Traffic Analysis GIF](Running_Real-Time_Traffic_Analysis.gif) 
# BADAL
