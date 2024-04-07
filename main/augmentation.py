import pandas as pd
import numpy as np

import os

from pathlib import Path
import shutil
from tqdm.auto import tqdm
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

import albumentations as A
import cv2

# removes all files and folders from dataset directory
def clear_directory(dir_path):    
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


## creates subsets of main dataset
def create_dataset(dataset_name, filtered_train_df, filtered_valid_df, class_names=['aircraft'], dataset_dir="datasets/"):
    dataset_dir = Path(dataset_dir) / dataset_name
    images_dir = dataset_dir / 'images'
    labels_dir = dataset_dir / 'labels'

    # Clear directories if they exist, to overwrite the dataset
    for subdir in ['train', 'valid']:
        img_subdir = images_dir / subdir
        label_subdir = labels_dir / subdir
        if img_subdir.exists():
            clear_directory(img_subdir)
        else:
            img_subdir.mkdir(parents=True, exist_ok=True)
        if label_subdir.exists():
            clear_directory(label_subdir)
        else:
            label_subdir.mkdir(parents=True, exist_ok=True)

    # Create directories
    for subdir in ['train', 'valid']:
        (images_dir / subdir).mkdir(parents=True, exist_ok=True)
        (labels_dir / subdir).mkdir(parents=True, exist_ok=True)

    def copy_file(src, dest):
        shutil.copy2(src, dest)

    def copy_files_concurrently(df, img_dest_dir, label_dest_dir):
        with ThreadPoolExecutor() as executor:
            # Prepare futures for image and label copying
            futures = [executor.submit(copy_file, row['image_path'], img_dest_dir / f"{Path(row['image_path']).name}") for _, row in df.iterrows()]
            futures += [executor.submit(copy_file, row['label_path'], label_dest_dir / f"{Path(row['label_path']).name}") for _, row in df.iterrows()]
            
            # Initialize progress bar
            pbar = tqdm(total=len(futures), desc='Copying files')
            for future in as_completed(futures):
                # Update progress bar upon task completion
                pbar.update(1)
            pbar.close()

    print("Copying training files:")
    copy_files_concurrently(filtered_train_df, images_dir / 'train', labels_dir / 'train')
    print("Copying validation files:")
    copy_files_concurrently(filtered_valid_df, images_dir / 'valid', labels_dir / 'valid')

    # Construct the YAML content with the desired structure
    yaml_content = {
        'path': str(f'../{dataset_dir}').replace('\\', '/'),  # Ensuring forward slashes
        'train': str('images/train'),
        'val': str('images/valid'),
        'names': {index: name for index, name in enumerate(class_names)}
    }  

    yaml_path = dataset_dir / f"{dataset_name}.yaml"
    with open(yaml_path, 'w') as file:
        yaml.dump(yaml_content, file, sort_keys=False)

    print(f"Dataset '{dataset_name}' created at {dataset_dir}")



# extracts class names and bboxes from all objects in label
def load_yolo_labels(label_path):
    with open(label_path, 'r') as file:
        labels = [line.strip().split() for line in file.readlines()]
        bboxes = [list(map(float, label[1:])) for label in labels]
        class_labels = [int(label[0]) for label in labels]
    return bboxes, class_labels

# takes class names and augmented bbox and converts into yolo label format
def format_yolo_label(class_labels, augmented_bboxes):
    label_str = ""
    for class_label, bbox in zip(class_labels, augmented_bboxes):
        label_str += f"{class_label} " + " ".join(f"{x:.6f}" for x in bbox) + "\n"
    return label_str



## horizontal/vertical flip

def augment_flip(image_path, label_path, orientation, p=1.0):
    # Load image - openCV
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load labels
    bboxes, class_labels = load_yolo_labels(label_path)
    
    # Define the augmentation based on the orientation parameter
    if orientation == 'h':
        flip = A.HorizontalFlip(p=p)
    elif orientation == 'v':
        flip = A.VerticalFlip(p=p)
    else:
        raise ValueError("Orientation must be 'h' or 'v'")
    
    transform = A.Compose([
        flip,
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    # Apply transformation
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    augmented_image = transformed['image']
    augmented_bboxes = transformed['bboxes']
    
    # YOLO formatted label: [class_id, x_center, y_center, width, height]
    augmented_label = format_yolo_label(class_labels, augmented_bboxes)
    
    return augmented_image, augmented_label


## rotation

def augment_rotation(image_path, label_path, angle, p=1.0):
   
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load labels
    bboxes, class_labels = load_yolo_labels(label_path)
    
    # Define the augmentation with rotation
    transform = A.Compose([
        A.Rotate(limit=(angle, angle), p=p, border_mode=cv2.BORDER_CONSTANT),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    # Apply transformation
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    augmented_image = transformed['image']
    augmented_bboxes = transformed['bboxes']
    
    # Convert augmented bboxes and class labels back to YOLO format
    augmented_label = format_yolo_label(class_labels, augmented_bboxes)
    
    return augmented_image, augmented_label

## contrast and brightness

# Alpha - contrast control (1.0-3.0)
# Beta - brightness control (-100 to 100)
def augment_brightness_contrast(image_path, alpha=1.0, beta=0):
    
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply brightness and contrast adjustment
    augmented_image = np.clip(alpha * image.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    
    return augmented_image

## histogram equalisation with CLAHE

def augment_histogram_equalization(image_path, p=1.0):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Define the augmentation
    transform = A.Compose([
        A.CLAHE(p),
    ])
    
    # Apply the augmentation
    transformed = transform(image=image)
    augmented_image = transformed['image']
    
    return augmented_image

## white balancing - gray word algorithm

def augmment_white_balance(image_path):
  
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Calculate the mean of each channel
    mr = np.mean(image[:, :, 0])
    mg = np.mean(image[:, :, 1])
    mb = np.mean(image[:, :, 2])
    
    # Calculate the overall mean
    mgray = (mr + mg + mb) / 3
    
    # Scale the channels based on the Gray World assumption
    image[:, :, 0] = np.clip(image[:, :, 0] * (mgray / mr), 0, 255)
    image[:, :, 1] = np.clip(image[:, :, 1] * (mgray / mg), 0, 255)
    image[:, :, 2] = np.clip(image[:, :, 2] * (mgray / mb), 0, 255)
    
    augmented_image = image.astype(np.uint8)
    return augmented_image

## sharpening 

def augment_sharpen(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Kernels from literature research
    kernel_1 = np.array([[0, -1, 0],
                         [-1, 5, -1],
                         [0, -1, 0]], dtype=np.float32)
    
    kernel_2 = np.array([[-1, -2, -1],
                         [-2, 13, -2],
                         [-1, -2, -1]], dtype=np.float32)

    kernel_2 = np.array([[-1, -2, -1],
                         [-2, 16, -2],
                         [-1, -2, -1]], dtype=np.float32)
    
    # Apply the sharpening kernel to the image
    sharpened_image = cv2.filter2D(image, -1, kernel_1)
    
    return sharpened_image

## guassian noise

def augment_gaussian_noise(image_path, var_limit=(10.0, 50.0)):
   
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Define the augmentation
    transform = A.Compose([
        A.GaussNoise(var_limit=var_limit, mean=0, p=1.0),
    ])
    
    # Apply the augmentation
    transformed = transform(image=image)
    augmented_image = transformed['image']
    
    return augmented_image

## zoom in 

# zoom factor: No zoom = 1, Full zoom (bounding box takes up entire picture) = 10
def augment_zoom(image_path, label_path, zoom_factor=1.5):

    # adjust zoom factor so that 1 = no zoom
    zoom_factor = zoom_factor / 10

    # Load image and labels
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes, class_labels = load_yolo_labels(label_path)

    # Calculate the crop dimensions based on the bounding box and zoom factor
    x_center, y_center, bbox_width, bbox_height = bboxes[0]  # Assuming one object
    x_center *= image.shape[1]  # Convert from relative to absolute coordinates
    y_center *= image.shape[0]
    bbox_width *= image.shape[1]
    bbox_height *= image.shape[0]

    # Define the crop dimensions
    crop_width = int(bbox_width / zoom_factor)
    crop_height = int(bbox_height / zoom_factor)
    
    # Calculate the crop coordinates
    x_min = max(0, int(x_center - crop_width / 2))
    y_min = max(0, int(y_center - crop_height / 2))
    x_max = min(image.shape[1], int(x_center + crop_width / 2))
    y_max = min(image.shape[0], int(y_center + crop_height / 2))

    # Define Albumentations transform for cropping and resizing
    transform = A.Compose([
        A.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, p=1.0),
        A.Resize(height=image.shape[0], width=image.shape[1], p=1.0)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    # Apply transformation
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    
    # The transformed bboxes are already in YOLO format
    augmented_label = format_yolo_label(class_labels, transformed_bboxes)

    return transformed_image, augmented_label


#method to save image
def save_image(path, image):

    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    

    # Convert from RGB to BGR
    image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Save the image
    cv2.imwrite(path, image_to_save)


# ethod to save label
def save_label(path, contents):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Now, save the file
    with open(path, 'w') as label_file:
        label_file.write(contents)