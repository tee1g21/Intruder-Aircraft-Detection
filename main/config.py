"""
Configuration for the Intruder-Aircraft-Detection project.

This configuration file defines paths and project-specific settings used across various parts of the project, 
including the dataset and logging directories for YOLO and custom classifier implementations.

Global Settings:
- BASE_DATASET: The base directory where the AVOIDDS dataset is located.
- PROJECT_NAME: The general name of the project for ClearML.

YOLO-specific settings:
- YOLO_DATASET_DIR: Directory for the YOLO formatted dataset.
- YOLO_PROJECT_DIR: Directory for logs and output related to the YOLO model training.
- YOLO_PROJECT_NAME: Name of the YOLO project within the main project, for ClearML.
- YOLO_CLASS_NAMES: List of class names specific to the YOLO model training.

Custom classifier settings:
- CLF_DATASET_DIR: Directory for the dataset formatted for use with a custom classifier.
- CLF_PROJECT_DIR: Directory for logs and output related to the custom classifier training.
- CLF_CLASS_NAMES: List of class names specific to the custom classifier training.
- CLF_PROJECT_NAME: Name of the custom classifier project within the main project, for ClearML.

"""

# Project variables
BASE_DATASET = 'C:/github/Third-Year-Project/Intruder-Aircraft-Detection/datasets/AVOIDDS'
PROJECT_NAME = 'Intruder-Aircraft-Detection'

# YOLO specfic
YOLO_DATASET_DIR = 'C:/github/Third-Year-Project/Intruder-Aircraft-Detection/datasets/YOLOv8'
YOLO_PROJECT_DIR = 'C:/github/Third-Year-Project/Intruder-Aircraft-Detection/logs/YOLOv8'
YOLO_PROJECT_NAME = f'{PROJECT_NAME}/YOLOv8'
YOLO_CLASS_NAMES = ['Cessna Skyhawk','Boeing 737-800', 'King Air C90'] 

# Custom classifier specific
CLF_DATASET_DIR = 'C:/github/Third-Year-Project/Intruder-Aircraft-Detection/datasets/Custom'
CLF_PROJECT_DIR = 'C:/github/Third-Year-Project/Intruder-Aircraft-Detection/logs/Custom'
CLF_CLASS_NAMES = ['Cessna Skyhawk','Boeing 737-800', 'King Air C90'] 
CLF_PROJECT_NAME = f'{PROJECT_NAME}/tl_clf'

