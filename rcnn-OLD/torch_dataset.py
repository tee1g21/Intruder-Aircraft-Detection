"""Python Class which takes augmented dataset originally in YOLO format and creates new Pytorch Dataset"""

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
import numpy as np
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
        try:
            # Load image
            img_path = self.images[idx]
            image = Image.open(img_path).convert('RGB') # Load the image and convert to RGB
            
            # Load bounding box labels
            label_path = os.path.join(self.labels_dir, os.path.basename(img_path).replace('.jpg', '.txt'))
            boxes = []
            labels = []
            with open(label_path, 'r') as file:
                for line in file:
                    class_label, x_center, y_center, width, height = map(float, line.split())
                    # Convert YOLO format to bounding box in pixel coordinates
                    x_min = (x_center - width / 2)
                    y_min = (y_center - height / 2)
                    x_max = (x_center + width / 2)
                    y_max = (y_center + height / 2)
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(int(class_label))

            # If you have transformations that need to be applied to the image, do them here
            if self.transform:
                image = self.transform(image)
            else:
                # Convert image to tensor if no other transformations are applied
                image = to_tensor(image)

            # Assuming all transformations are scale-invariant or properly rescale boxes
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

            # Convert bounding boxes from relative to absolute coordinates
            img_width, img_height = image.shape[2], image.shape[1]  # Tensor shape is C x H x W
            boxes[:, [0, 2]] *= img_width
            boxes[:, [1, 3]] *= img_height

            # Create target dictionary
            target = {
                'boxes': boxes,  # Ensure the boxes are correctly formatted as [x_min, y_min, x_max, y_max]
                'labels': labels,
                'image_id': torch.tensor([idx])  # Single-element tensor
            }
            
            area = (target['boxes'][:, 3] - target['boxes'][:, 1]) * (target['boxes'][:, 2] - target['boxes'][:, 0])
            target['area'] = area


            return image, target
        except Exception as e:
            print(f"Error processing file: {self.images[idx]}")
            raise RuntimeError(f"Failed due to {e}")
 
        

