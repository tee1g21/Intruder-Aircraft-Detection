"""Python file containing constant project variables which will be used in testing but can only be changed in this file"""

# Project variables
BASE_DATASET = 'C:/github/Third-Year-Project/Intruder-Aircraft-Detection/datasets/AVOIDDS'
#BASE_DATASET = '/mnt/c/github/Third-Year-Project/Intruder-Aircraft-Detection/datasets/AVOIDDS'
PROJECT_NAME = 'Intruder-Aircraft-Detection'

# YOLO specfic
YOLO_DATASET_DIR = 'C:/github/Third-Year-Project/Intruder-Aircraft-Detection/datasets/YOLOv8'
YOLO_PROJECT_DIR = 'C:/github/Third-Year-Project/Intruder-Aircraft-Detection/logs/YOLOv8'
YOLO_PROJECT_NAME = f'{PROJECT_NAME}/YOLOv8'
#YOLO_CLASS_NAMES = ['aircraft']
YOLO_CLASS_NAMES = ['Cessna Skyhawk','Boeing 737-800', 'King Air C90'] # TODO: give simpler class name (eg. cessna, boeing, king-air)

# Custom classifier specific
CLF_DATASET_DIR = 'C:/github/Third-Year-Project/Intruder-Aircraft-Detection/datasets/Custom'
CLF_PROJECT_DIR = 'C:/github/Third-Year-Project/Intruder-Aircraft-Detection/logs/Custom'
#CLF_DATASET_DIR = '/mnt/c/github/Third-Year-Project/Intruder-Aircraft-Detection/datasets/Custom'
#CLF_PROJECT_DIR = '/mnt/c/github/Third-Year-Project/Intruder-Aircraft-Detection/logs/Custom'
CLF_CLASS_NAMES = ['Cessna Skyhawk','Boeing 737-800', 'King Air C90'] # TODO: give simpler class name (eg. cessna, boeing, king-air)
CLF_PROJECT_NAME = f'{PROJECT_NAME}/tl_clf'

# RCNN specific
#RCNN_DATASET_DIR = 'C:/github/Third-Year-Project/Intruder-Aircraft-Detection/datasets/RCNN'
#RCNN_PROJECT_DIR = 'C:/github/Third-Year-Project/Intruder-Aircraft-Detection/logs/RCNN'
#RCNN_CLASS_NAMES = YOLO_CLASS_NAMES # keep the same for safe testing
