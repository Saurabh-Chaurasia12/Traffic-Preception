import torch

# File paths
DATA_PATH = 'data/'
MODEL_PATH = 'models/yolov8n.pt'
WEIGHTS_PATH = 'weights/yolov8n.pt'
OPTIMIZED_WEIGHTS_PATH = 'weights/yolov8n_optimized.trt'

# Model parameters
NUM_EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 0.005

# Other settings
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
