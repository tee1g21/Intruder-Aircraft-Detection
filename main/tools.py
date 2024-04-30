"""Python file containing helper methods for project"""

import os
import shutil
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches




# remove directories if they exist
def remove_if_exists(path, progress=None):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.isfile(path):
        os.remove(path)    
    if progress:
        progress.update(1)

# save image and label
def save_image(path, image):

    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)    
    # Convert from RGB to BGR
    image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)    
    # Save the image
    cv2.imwrite(path, image_to_save)
def save_label(path, contents):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)    
    # Now, save the file
    with open(path, 'w') as label_file:
        label_file.write(contents)

# Copies individual file from src to dst
def copy_file(src, dst):
    shutil.copy2(src, dst)

# moves individual file from src to dst   
def move_file(src, dst):
    shutil.move(src,dst)

# Extracts class names and bboxes from all objects in label
def load_yolo_labels(label_path):
    with open(label_path, 'r') as file:
        labels = [line.strip().split() for line in file.readlines()]
        bboxes = [list(map(float, label[1:])) for label in labels]
        class_labels = [int(label[0]) for label in labels]
    return bboxes, class_labels

# Takes class names and augmented bbox and converts into yolo label format
def format_yolo_label(class_labels, augmented_bboxes):
    label_str = ""
    for class_label, bbox in zip(class_labels, augmented_bboxes):
        label_str += f"{class_label} " + " ".join(f"{x:.6f}" for x in bbox) + "\n"
    return label_str


# TEST - overlay bboxes on images
def overlay_bbox_image(image_path, label_path):

    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load labels
    bboxes, class_labels = load_yolo_labels(label_path)
    
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    height, width, _ = image.shape
    for bbox, class_label in zip(bboxes, class_labels):
        x_center, y_center, bbox_width, bbox_height = bbox
        x_min = (x_center - bbox_width / 2) * width
        y_min = (y_center - bbox_height / 2) * height
        
        rect = patches.Rectangle((x_min, y_min), bbox_width * width, bbox_height * height,
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x_min, y_min - 2, str(class_label), color='red', fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='red', boxstyle='round'))
    
    plt.axis('off')
    plt.show()
    
# converts dictionary to pretty string
def pretty_print_dict(d, indent=0):
        lines = []
        # Create indentation spaces
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
            elif isinstance(value, (str, float, int, bool)):
                # Print strings, floats, ints, and bools on the same line as the key
                lines[-1] += f" {value}"
            else:
                # If it's not a recognized type, print the type name
                lines.append(f"{indent_space}    Unknown type: {type(value)}")
        return "\n".join(lines)