"""Tools module for various utility functions used throughout the project.

This module provides a collection of utility functions that perform common tasks such as file handling, 
label formatting, and metric calculations.
"""

import os
import shutil
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import numpy as np

# remove directories if they exist
def remove_if_exists(path, progress=None):
    """
    Removes a file or directory if it exists.

    Parameters:
        path (str): Path to the file or directory to be removed.
        progress (tqdm, optional): A tqdm progress bar to update upon completion; default is None.

    This function checks if the specified path is a file or directory and removes it if it exists.
    If a progress bar object is provided, it updates the progress bar after the removal.
    """
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.isfile(path):
        os.remove(path)    
    if progress:
        progress.update(1)

# save image and label
def save_image(path, image):
    """
    Saves an image to the specified path, ensuring the directory exists.

    Parameters:
        path (str): Path where the image will be saved.
        image (numpy.ndarray): Image data in RGB format to be saved.

    This function converts the image from RGB to BGR format and saves it to the specified path.
    If the directory does not exist, it creates the necessary directories.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)    
    # Convert from RGB to BGR
    image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)    
    # Save the image
    cv2.imwrite(path, image_to_save)
    
def save_label(path, contents):
    """
    Saves label contents to the specified path, ensuring the directory exists.

    Parameters:
        path (str): Path where the label file will be saved.
        contents (str): The content to be written to the label file.

    This function ensures the directory exists and then saves the provided content to the specified path.
    If the directory does not exist, it creates the necessary directories.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)    
    # Now, save the file
    with open(path, 'w') as label_file:
        label_file.write(contents)

# Copies individual file from src to dst
def copy_file(src, dst):
    """
    Copies a file from a source path to a destination path.

    Parameters:
        src (str): The source file path.
        dst (str): The destination file path.

    This function copies the file located at the source path to the destination path, preserving metadata.
    """
    shutil.copy2(src, dst)

# moves individual file from src to dst   
def move_file(src, dst):
    """
    Moves a file from a source path to a destination path.

    Parameters:
        src (str): The source file path.
        dst (str): The destination file path.

    This function moves the file located at the source path to the destination path.
    """
    shutil.move(src,dst)

# Extracts class names and bboxes from all objects in label
def load_yolo_labels(label_path):
    """
    Loads YOLO-formatted labels from a file.

    Parameters:
        label_path (str): Path to the label file.

    Returns:
        tuple: A tuple containing:
            - bboxes (list of list of float): List of bounding boxes with coordinates.
            - class_labels (list of int): List of class labels.

    This function reads a YOLO-formatted label file, extracting bounding box coordinates and class labels,
    and returns them as separate lists.
    """
    with open(label_path, 'r') as file:
        labels = [line.strip().split() for line in file.readlines()]
        bboxes = [list(map(float, label[1:])) for label in labels]
        class_labels = [int(label[0]) for label in labels]
    return bboxes, class_labels

# Takes class names and augmented bbox and converts into yolo label format
def format_yolo_label(class_labels, augmented_bboxes):
    """
    Formats class labels and bounding boxes into YOLO label format.

    Parameters:
        class_labels (list of int): List of class labels.
        augmented_bboxes (list of list of float): List of augmented bounding boxes.

    Returns:
        str: A formatted string representing the YOLO labels.

    This function takes class labels and bounding boxes, formats them into YOLO label format, 
    and returns the resulting string.
    """
    label_str = ""
    for class_label, bbox in zip(class_labels, augmented_bboxes):
        label_str += f"{class_label} " + " ".join(f"{x:.6f}" for x in bbox) + "\n"
    return label_str
    
# converts dictionary to pretty string
def pretty_print_dict(d, indent=0):
    """
    Formats a dictionary into a human-readable string with indentation.

    Parameters:
        d (dict): The dictionary to be formatted.
        indent (int, optional): The number of spaces to use for indentation (default is 0).

    Returns:
        str: A formatted string representing the dictionary.

    This function takes a dictionary and formats it into a human-readable string with indentation.
    It handles nested dictionaries and lists, formatting them appropriately.
    """
    lines = []
    indent_space = ' ' * indent
    for key, value in d.items():
        # Add the key with a colon
        lines.append(f"{indent_space}{key}:")
        if isinstance(value, dict):
            # Recursively format nested dictionaries with increased indent
            lines.append(pretty_print_dict(value, indent=indent+2))
        elif isinstance(value, list):
            # Format list items with a hyphen and increased indent
            for item in value:
                lines.append(f"{indent_space}    - {item}")
        elif isinstance(value, np.ndarray):
            for item in value:
                lines.append(f"{indent_space}    - {item}")
        elif isinstance(value, (str, float, int, bool)):
            # Print strings, floats, ints, and bools on the same line as the key
            lines[-1] += f" {value}"
        else:
            # If it's not a recognized type, print the type name
            lines.append(f"{indent_space}    Unknown type: {type(value)}")
    return "\n".join(lines)

# counts images in keras dataset directory
def count_images(directory):
    """
    Counts the total number of files in a directory, including all subdirectories.

    Parameters:
        directory (str): The directory in which to count files.

    Returns:
        int: The total number of files in the directory.

    This function traverses the specified directory and all its subdirectories, 
    counting the total number of files.
    """
    total_files = 0
    for root, dirs, files in os.walk(directory):
        total_files += len(files)
    return total_files

# generate unique seed 
def generate_seed():
    """
    Generates a seed value based on the current time in milliseconds.

    Returns:
        int: A seed value derived from the current time.

    This function calculates the current time in milliseconds and returns it as a seed value,
    modulo 2^32 to ensure it fits within typical integer ranges used for seeds.
    """
    # Get current time in milliseconds
    milliseconds = int(round(time.time() * 1000))
    return milliseconds % (2**32)

# round to 3 significant figures
def round_to_3sf(value):
    """
    Rounds a value or each element of a collection to three significant figures.

    Parameters:
        value (float, list, or np.ndarray): The value or collection of values to round.

    Returns:
        float, list, or np.ndarray: The rounded value or collection of rounded values.

    This function rounds a given float, or each element in a list or numpy array, to three significant figures.
    """
    if isinstance(value, np.ndarray):
        return np.vectorize(lambda x: float(f"{x:.3g}"))(value)  # Apply rounding using a vectorized function
    elif isinstance(value, list):
        return [float(f"{v:.3g}") for v in value]
    else:
        return float(f"{value:.3g}")