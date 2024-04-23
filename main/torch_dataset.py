"""Python Class which takes augmented dataset originally in YOLO format and creates new Pytorch Dataset"""


import torch
from torch.utils.data import Dataset

import os
import yaml
from PIL import Image

class TorchDataset(Dataset):
    def __init__(self, dataset_dir, subset, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.images_dir = os.path.join(self.dataset_dir, 'images', subset)
        self.labels_dir = os.path.join(self.dataset_dir, 'labels', subset)
        self.images = [os.path.join(self.images_dir, img) for img in sorted(os.listdir(self.images_dir)) if img.endswith('.jpg')]
        self.dataset_name = os.path.basename(self.dataset_dir)
        
        # Load class names from the YAML file
        yaml_path = os.path.join(self.dataset_dir, f'{self.dataset_name}.yaml')
        with open(yaml_path) as f:
            self.class_names = yaml.safe_load(f)['names']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')  # Load the image and convert to RGB
        orig_width, orig_height = image.width, image.height  # Original image size

        # Load bounding box labels
        label_path = os.path.join(self.labels_dir, os.path.basename(img_path).replace('.jpg', '.txt'))
        boxes = []
        labels = []
        with open(label_path, 'r') as file:
            for line in file:
                class_label, x_center, y_center, width, height = map(float, line.split())
                # Convert YOLO format to bounding box in pixel coordinates
                x_min = (x_center - width / 2) * orig_width
                y_min = (y_center - height / 2) * orig_height
                x_max = (x_center + width / 2) * orig_width
                y_max = (y_center + height / 2) * orig_height
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(int(class_label))

        # If you have transformations that need to be applied to the image, do them here
        if self.transform:
            image = self.transform(image)

        # Convert image to tensor after transformations
        new_width, new_height = image.shape[2], image.shape[1]  # Tensor shape is C x H x W

        # Calculate scaling factors separately for width and height
        scale_x = new_width / orig_width
        scale_y = new_height / orig_height

        # Apply scaling factors to the bounding boxes
        scaled_boxes = []
        for box in boxes:
            scaled_boxes.append([
                box[0] * scale_x, 
                box[1] * scale_y, 
                box[2] * scale_x, 
                box[3] * scale_y
            ])
        scaled_boxes = torch.tensor(scaled_boxes, dtype=torch.float32)

        # Create target dictionary
        target = {
            'boxes': scaled_boxes, 
            'labels': torch.tensor(labels, dtype=torch.int64), 
            'image_id': torch.tensor([idx])
        }

        return image, target