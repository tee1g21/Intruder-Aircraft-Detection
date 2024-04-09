"""Python file containing augmentation methods to perform augmentation on individual images (and labels)"""

import Tools
import cv2
import albumentations as A
import numpy as np
import os

# horizontal/vertical flip
def flip(image_path, label_path, orientation, p=1.0):
    
    # Load image and labels
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes, class_labels = Tools.load_yolo_labels(label_path)
    
    # Define the augmentation from orientation parameter
    if orientation == 'h':
        flip = A.HorizontalFlip(p=p)
    elif orientation == 'v':
        flip = A.VerticalFlip(p=p)
    else:
        raise ValueError("Orientation must be 'h' or 'v'")
    
    # Compose transform
    transform = A.Compose([
        flip,
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    # Apply transformation
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    augmented_image = transformed['image']
    augmented_bboxes = transformed['bboxes']
    
    # YOLO formatted label: [class_id, x_center, y_center, width, height]
    augmented_label = Tools.format_yolo_label(class_labels, augmented_bboxes)
    
    return augmented_image, augmented_label

# rotation with specified angle
def rotate(image_path, label_path, angle, p=1.0):
   
    # Load image and label
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
    bboxes, class_labels = Tools.load_yolo_labels(label_path)
    
    # Define the augmentation with rotation
    transform = A.Compose([
        A.Rotate(limit=(angle, angle), p=p, border_mode=cv2.BORDER_CONSTANT),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    # Apply transformation
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    augmented_image = transformed['image']
    augmented_bboxes = transformed['bboxes']    

    # Convert augmented bboxes and class labels back to YOLO format
    augmented_label = Tools.format_yolo_label(class_labels, augmented_bboxes)
    
    return augmented_image, augmented_label

# contrast and brightness
def brightness_and_contrast(image_path, alpha=1.0, beta=0):
    """
    Alpha - contrast control (1.0-3.0)
    Beta - brightness control (-100 to 100)
    """

    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply brightness and contrast adjustment
    augmented_image = np.clip(alpha * image.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    
    return augmented_image

# histogram equalisation with CLAHE
def hist_eq(image_path, p=1.0):
    
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

# white balancing - gray word algorithm
def white_balance(image_path):
  
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

# kernel sharpening
def sharpen(image_path):
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

# guassian noise
def gaussian_noise(image_path, var_limit=(10.0, 50.0)):
   
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

# zoom centred about object
def zoom(image_path, label_path, zoom_factor=1):
    """
    zoom factor: No zoom = 1, Full zoom (bounding box takes up entire picture) = 10
    """
    # adjust zoom factor so that 1 = no zoom
    zoom_factor = zoom_factor / 10

    # Load image and labels
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes, class_labels = Tools.load_yolo_labels(label_path)

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
    augmented_image = transformed['image']
    augmented_bboxes = transformed['bboxes']
    
    # The transformed bboxes are already in YOLO format
    augmented_label = Tools.format_yolo_label(class_labels, augmented_bboxes)

    return augmented_image, augmented_label

# shift hue, saturation or value or HSV
def hsv_shift(image_path, type, shift, p=1.0):
    
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Prepare shift values based on the type
    hue_shift = shift if type == 'h' else 0
    sat_shift = shift if type == 's' else 0
    val_shift = shift if type == 'v' else 0

    # Define the augmentation pipeline
    transform = A.Compose([
        A.HueSaturationValue(
            hue_shift_limit=(hue_shift, hue_shift),  # Setting both limits to the same value for exact shift
            sat_shift_limit=(sat_shift, sat_shift),  
            val_shift_limit=(val_shift, val_shift),  
            p=p
        )
    ])

    # Apply the augmentation
    transformed = transform(image=image)
    augmented_image = transformed['image']
    
    return augmented_image

# augments and saves individual image and label
def augment_image(image_path, images_aug_dir, label_path, labels_aug_dir, method_name, method_info):
    
    # raw name of image and label without file extension
    image_name, _ = os.path.splitext(os.path.basename(image_path))
    label_name, _ = os.path.splitext(os.path.basename(label_path))

    # throw error if image does not match label
    if image_name != label_name:
        raise ValueError(f"ERROR: Filename mismatch: {image_name} and {label_name} do not match.")

    # create filenames for augmented images
    aug_image_filename = f"{image_name}-{method_name}.jpg"
    aug_label_filename = f"{label_name}-{method_name}.txt"
    
    # Determine which augmentation function to call based on the method name
    if method_name == 'flip':
        augmented_image, augmented_label = flip(image_path, label_path, **method_info['parameters'])
    elif method_name == 'rotate':
        augmented_image, augmented_label = rotate(image_path, label_path, **method_info['parameters'])
    elif method_name == 'bnc':
        augmented_image = brightness_and_contrast(image_path, **method_info['parameters'])
        augmented_label = open(label_path).read()  # No change to label for this augmentation
    elif method_name == 'gaussian':
        augmented_image = gaussian_noise(image_path, **method_info['parameters'])
        augmented_label = open(label_path).read()  # No change to label for this augmentation
    elif method_name == 'histEq':
        augmented_image = hist_eq(image_path, **method_info['parameters'])
        augmented_label = open(label_path).read()  # No change to label for this augmentation
    elif method_name == 'whiteBal':
        augmented_image = white_balance(image_path)
        augmented_label = open(label_path).read()  # No change to label for this augmentation
    elif method_name == 'sharpen':
        augmented_image = sharpen(image_path)
        augmented_label = open(label_path).read()  # No change to label for this augmentation
    elif method_name == 'zoom':
        augmented_image, augmented_label = zoom(image_path, label_path, **method_info['parameters'])
    elif method_name == 'hsv':
        augmented_image = hsv_shift(image_path, **method_info['parameters'])
        augmented_label = open(label_path).read()  # No change to label for this augmentation
    
    # Save the augmented image and label
    Tools.save_image(images_aug_dir + aug_image_filename, augmented_image)
    Tools.save_label(labels_aug_dir + aug_label_filename, augmented_label)
