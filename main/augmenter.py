"""
This module provides a collection of image augmentation functions. The augmentations include 
operations like flipping, rotating, adjusting brightness and contrast, applying Gaussian noise,
histogram equalization, white balancing, sharpening, and zooming. 

Functions:
    flip(image_path, label_path, orientation, p=1.0): Applies a flip transformation.
    rotate(image_path, label_path, angle_limit, p=1.0): Applies a rotation to the image.
    brightness_and_contracy(image_path, alpha=1.0, beta=0): Modifies brightness and contrast.
    gaussian_noise(image_path, var_limit=(10.0, 50.0)): Adds Gaussian noise to images.
    hist_eq(image_path, p=1.0): Applies histogram equalization for contrast improvement.
    white_balance(image_path): Adjusts image colors based on the Gray World assumption.
    sharpen(image_path): Enhances image sharpness using a kernel-based approach.
    zoom(image_path, label_path, zoom_factor=1): Zooms into the image focusing on central objects.
    augment_image(image_path, images_aug_dir, label_path, labels_aug_dir, method_name, method_info):
        General method to apply specified augmentations and save the outputs.

Utilities:
    - OpenCV: Used for image manipulation.
    - Albumentations: Utilized for applying complex transformations.
    - Numpy: Required for array operations.
    
"""

import tools
import cv2
import albumentations as A
import numpy as np
import os
import random
import uuid

# horizontal/vertical flip
def flip(image_path, label_path, orientation, p=1.0):
    """
    Performs a horizontal or vertical flip transformation on an image based on the specified orientation, 
    and accordingly updates the associated bounding box labels for object detection.

    Parameters:
        image_path (str): Path to the image file.
        label_path (str): Path to the label file.
        orientation (Jupyter Notebook (Python)): 'h' for horizontal, 'v' for vertical.
        p (float): Probability of the flip.

    Returns:
        tuple: Augmented image and updated labels in the same format as input.
    """    
    
    # Load image and labels
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes, class_labels = tools.load_yolo_labels(label_path)
    
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
    augmented_label = tools.format_yolo_label(class_labels, augmented_bboxes)
    
    return augmented_image, augmented_label

# rotation with specified angle
def rotate(image_path, label_path, angle_limit, p=1.0):
    """Rotates an image within a specified angle range and updates its bounding box labels. 
    The rotation is random within the specified limits and may be applied with a certain probability.

    Parameters:
        image_path (str): Path to the image file.
        label_path (str): Path to the label file.
        angle_limit (int): Maximum degree to rotate the image. Rotation will be between -angle_limit and +angle_limit.
        p (float): Probability of applying the rotation.

    Returns:
        tuple: Augmented image and updated labels in YOLO format.
    """   
   
    # Load image and label
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes, class_labels = tools.load_yolo_labels(label_path)
    
    # Define the augmentation with random rotation
    transform = A.Compose([
        A.Rotate(limit=(-angle_limit, angle_limit), p=p, border_mode=cv2.BORDER_CONSTANT),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    # Apply transformation
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    augmented_image = transformed['image']
    augmented_bboxes = transformed['bboxes']    

    # Convert augmented bboxes and class labels back to YOLO format
    augmented_label = tools.format_yolo_label(class_labels, augmented_bboxes)
    
    return augmented_image, augmented_label

# contrast and brightness
def brightness_and_contrast(image_path, alpha=1.0, beta=0):
    """
    Adjusts the brightness and contrast of an image using specified alpha and beta values. 
    Alpha controls the contrast and can vary between 1.0 and 3.0. Beta adjusts the brightness and can range from -100 to 100.

    Parameters:
        image_path (str): Path to the image file.
        alpha (float): Factor by which to scale the contrast (default 1.0).
        beta (int): Value to adjust the brightness (default 0).

    Returns:
        numpy.ndarray: Image array with adjusted brightness and contrast.
    """
    
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Generate random alpha and beta within the limits
    rnd_alpha = random.uniform(1.0, alpha)  # Contrast control between 1.0 and alpha_limit
    rnd_beta = random.uniform(0, alpha)     # Brightness control between 0 and beta_limit

    # Apply brightness and contrast adjustment
    augmented_image = np.clip(rnd_alpha * image.astype(np.float32) + rnd_beta, 0, 255).astype(np.uint8)

    return augmented_image

# histogram equalisation with CLAHE
def hist_eq(image_path, p=1.0):
    """
    Applies histogram equalization to an image using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to improve image contrast. This operation is applied with a specified probability.

    Parameters:
        image_path (str): Path to the image file.
        p (float): Probability of applying the histogram equalization (default 1.0).

    Returns:
        numpy.ndarray: Image array with improved contrast.
    """
    
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
    """
    Applies white balance to an image using the Gray World assumption. 
    This technique adjusts the colors of the image so that the average color is neutral gray.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        numpy.ndarray: Image array with applied white balance.
    """   
  
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
    """
    Applies a sharpening filter to an image to enhance its edges and details.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        numpy.ndarray: Image array with applied sharpening.
    """
    
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Kernels from literature research
    kernel_1 = np.array([[0, -1, 0],
                         [-1, 5, -1],
                         [0, -1, 0]], dtype=np.float32)
    
    # Apply the sharpening kernel to the image
    sharpened_image = cv2.filter2D(image, -1, kernel_1)
    
    return sharpened_image

# guassian noise
def gaussian_noise(image_path, var_limit=(10.0, 50.0)):
    """
    Applies Gaussian noise to an image based on a specified variance range. 

    Parameters:
        image_path (str): Path to the image file.
        var_limit (tuple): Minimum and maximum variance for the noise.

    Returns:
        numpy.ndarray: Image array with applied Gaussian noise.
    """
   
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
    Applies a zoom operation on an image based on a specified zoom factor and updates the corresponding
    bounding boxes. The function adjusts the view to focus on the primary object and resizes the crop to the original image size.

    Parameters:
        image_path (str): Path to the image file.
        label_path (str): Path to the file containing YOLO formatted bounding box labels.
        zoom_factor (int): Control for zoom intensity. 0 means no zoom, up to 10 for maximum zoom, 
                           where the object takes up the entire image.

    Returns:
        tuple: Tuple containing the zoomed and resized image and the updated labels, both in their original dimensions and format.
    """
    # adjust zoom factor so that 1 = no zoom
    zoom_factor = zoom_factor / 10

    # Load image and labels
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes, class_labels = tools.load_yolo_labels(label_path)

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
    augmented_label = tools.format_yolo_label(class_labels, augmented_bboxes)

    return augmented_image, augmented_label


# augments and saves individual image and label
def augment_image(image_path, images_aug_dir, label_path, labels_aug_dir, method_name, method_info):
    """
    Applies a specified image augmentation method to an image and saves the augmented image and label. 
    Handles various augmentation techniques such as flipping, rotating, adjusting brightness and contrast, 
    adding noise, and more. Each method may adjust the image, label, or both, depending on its nature.

    Parameters:
        image_path (str): Path to the original image.
        images_aug_dir (str): Directory path to save augmented images.
        label_path (str): Path to the original label file.
        labels_aug_dir (str): Directory path to save augmented labels.
        method_name (str): Name of the augmentation method to apply.
        method_info (dict): Contains parameters specific to the chosen augmentation method.

    Raises:
        ValueError: If the filenames of the image and label do not match.

    Notes:
        Augmentation methods that do not modify labels will simply copy the original label.
    """    
    
    # raw name of image and label without file extension
    image_name, _ = os.path.splitext(os.path.basename(image_path))
    label_name, _ = os.path.splitext(os.path.basename(label_path))

    # throw error if image does not match label
    if image_name != label_name:
        raise ValueError(f"ERROR: Filename mismatch: {image_name} and {label_name} do not match.")

    # Generate a unique identifier for each augmented file
    unique_id = uuid.uuid4().hex[:8]

    # create filenames for augmented images
    aug_image_filename = f"{image_name}-{method_name}-{unique_id}.jpg"
    aug_label_filename = f"{label_name}-{method_name}-{unique_id}.txt"
    
    # Determine which augmentation function to call based on the method name
    if method_name == 'flip':
        augmented_image, augmented_label = flip(image_path, label_path, **method_info['parameters'])
    elif method_name == 'rotate':
        augmented_image, augmented_label = rotate(image_path, label_path, **method_info['parameters'])
    elif method_name == 'rotate2':
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
    
    # Save the augmented image and label
    tools.save_image(images_aug_dir + aug_image_filename, augmented_image)
    tools.save_label(labels_aug_dir + aug_label_filename, augmented_label)
