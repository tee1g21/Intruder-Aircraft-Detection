"""Python File containing methods to handle the main dataset, create sub datasets and to augment these datasets"""

import Tools
import Augmenter

import os
import pandas as pd
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
max_workers=12
from tqdm.auto import tqdm
import yaml
import random

# Function to create a DataFrame from images and labels
def create_dataframe(images_path, labels_path, metadata_path):

    # Load metadata
    with open(metadata_path, 'r') as file:
        metadata = json.load(file)

    # List all files in the directories
    image_files = [f for f in sorted(os.listdir(images_path)) if f.endswith('.jpg')]
    label_files = [f for f in sorted(os.listdir(labels_path)) if f.endswith('.txt')]
    
    # Create tempory DataFrame so that final dataframe is in correct order
    temp_df = pd.DataFrame({
        'image_path': [str(images_path + '/' + file) for file in image_files],
        'label_path': [str(labels_path + '/' + file) for file in label_files],
    })

    # Extract image indices to match with metadata
    df = pd.DataFrame()
    df['imageID'] = temp_df['image_path'].apply(lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    # Add image and label paths to final dataframe
    df['image_path'] = temp_df['image_path']
    df['label_path'] = temp_df['label_path']
 
    # Add metadata to each image entry
    for key, value in metadata.items():
        if '.' in key:  # Key represents a range
            start, end = map(int, key.split('.'))
            df.loc[df['imageID'].between(start, end), 'metadata'] = json.dumps(value)

    # Convert the JSON strings in 'metadata' to dictionaries
    df['metadata'] = df['metadata'].apply(json.loads)

    # Expand the 'metadata' column into separate columns
    metadata_df = pd.json_normalize(df['metadata'])
    
    # Concatenate the expanded metadata back to the original DataFrame
    full_df = pd.concat([df.drop(['metadata'], axis=1), metadata_df], axis=1)

    return full_df

# Copies files from dataframe to new dataset - with multithreading
def copy_dataframe_files_concurrently(df, img_dest_dir, label_dest_dir):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Prepare futures for image and label copying
            futures = [executor.submit(Tools.copy_file, row['image_path'], img_dest_dir / f"{Path(row['image_path']).name}") for _, row in df.iterrows()]
            futures += [executor.submit(Tools.copy_file, row['label_path'], label_dest_dir / f"{Path(row['label_path']).name}") for _, row in df.iterrows()]
            
            # Initialize progress bar
            pbar = tqdm(total=len(futures), desc='Copying files')
            for future in as_completed(futures):
                # Update progress bar upon task completion
                pbar.update(1)
            pbar.close()

# creates subsets of main dataset
def create_sub_dataset(dataset_name, filtered_train_df, filtered_valid_df, class_names=['aircraft'], dataset_dir="C:/github/Third-Year-Project/Intruder-Aircraft-Detection/datasets"):
    new_dataset_dir = Path(dataset_dir) / dataset_name
    images_dir = Path(new_dataset_dir) / 'images'
    labels_dir = Path(new_dataset_dir) / 'labels'

    print("Removing dataset if pre-existing")
    Tools.remove_if_exists(new_dataset_dir)

    # Create directories
    for subdir in ['train', 'valid']:
        (images_dir / subdir).mkdir(parents=True, exist_ok=True)
        (labels_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Copy files from dataframe
    print("Copying training files:")
    copy_dataframe_files_concurrently(filtered_train_df, images_dir / 'train', labels_dir / 'train')
    print("Copying validation files:")
    copy_dataframe_files_concurrently(filtered_valid_df, images_dir / 'valid', labels_dir / 'valid')

    # Construct the YAML content with the desired structure
    yaml_content = {
        'path': str(f'{new_dataset_dir}').replace('\\', '/'),  # Ensuring forward slashes
        'train': str(f'{images_dir}/train').replace('\\', '/'),
        'val': str(f'{images_dir}/valid').replace('\\', '/'),
        'names': {index: name for index, name in enumerate(class_names)}
    }  

    yaml_path = new_dataset_dir / f"{dataset_name}.yaml"
    with open(yaml_path, 'w') as file:
        yaml.dump(yaml_content, file, sort_keys=False)

    print(f"Dataset '{dataset_name}' created at {dataset_dir}")

# creates augmented dataset structure
def create_augmented_dataset_structure(original_dataset_path):
    
    """
    Creates directory for augmented dataset:
     - adds a train-aug to images and labels (as well as train)
     - adds another yaml file with -aug appended which points to train-aug rather than train
    """

    # Create train-aug paths
    images_train_aug_path = original_dataset_path + '/images/train-aug'
    labels_train_aug_path = original_dataset_path + '/labels/train-aug'

    # get raw database name
    dataset_name, _ = os.path.splitext(os.path.basename(original_dataset_path))

    # Path for original and new yaml
    original_yaml_path = original_dataset_path + f'/{dataset_name}.yaml'
    augmented_yaml_path = original_dataset_path + f'/{dataset_name}-aug.yaml'

    paths_to_remove = [
        images_train_aug_path,
        labels_train_aug_path,
        augmented_yaml_path
    ]   

    # Replace existing directories
    with tqdm(total=len(paths_to_remove), desc="Removing existing directories/files") as progress:
        # Check and remove each path, updating progress
        for path in paths_to_remove:
            Tools.remove_if_exists(path, progress)

    # Create directories
    os.makedirs(images_train_aug_path, exist_ok=True)
    os.makedirs(labels_train_aug_path, exist_ok=True)
    
    # create new yaml with aug appended to train  
    with open(original_yaml_path, 'r') as file:
        dataset_config = yaml.safe_load(file) 

    dataset_config['train'] += "-aug"

    with open(augmented_yaml_path, 'w') as file:
        yaml.safe_dump(dataset_config, file, default_flow_style=False, sort_keys=False)

# copies all files to another directory using multithreading
def copy_directory_contents_concurrently(src_dir, dst_dir):
     
    # Retrieve a list of source file paths
    src_files = [os.path.join(src_dir, file_name) for file_name in os.listdir(src_dir)]
    dst_files = [os.path.join(dst_dir, os.path.basename(file_path)) for file_path in src_files]
    
    # Copy files concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(Tools.copy_file, src_files, dst_files), total=len(src_files), desc="Copying files"))

# Creates augmented dataset, using metadata dictionary to define augmentation methods
def augment_dataset(original_dataset_path, augmentation_metadata):
    
    # reconstruct dataset with augmentation directories and yaml
    create_augmented_dataset_structure(original_dataset_path)

    # new train directories
    images_dir = original_dataset_path + '/images/train/'
    images_aug_dir = original_dataset_path + '/images/train-aug/'
    labels_dir = original_dataset_path + '/labels/train/'
    labels_aug_dir = original_dataset_path + '/labels/train-aug/'

    # copy train and image labels to train-aug
    copy_directory_contents_concurrently(images_dir, images_aug_dir)
    copy_directory_contents_concurrently(labels_dir, labels_aug_dir)

    # training images 
    image_paths = [os.path.join(images_aug_dir, img_name) for img_name in os.listdir(images_aug_dir)]

    # Apply augmentations based on metadata
    total_augmentations = sum(int(len(image_paths) * info['apply_to_percentage']) for info in augmentation_metadata['methods'].values())
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for method_name, method_info in augmentation_metadata['methods'].items():
            selected_images = random.sample(image_paths, int(len(image_paths) * method_info['apply_to_percentage']))
            selected_labels = [path.replace(images_aug_dir, labels_aug_dir).replace('.jpg', '.txt') for path in selected_images]
            
            for image_path, label_path in zip(selected_images, selected_labels):
                # Schedule the augmentation to be applied concurrently
                futures.append(executor.submit(Augmenter.augment_image, image_path, images_aug_dir, label_path, labels_aug_dir, method_name, method_info))

        # Progress bar for the augmentation tasks
        for _ in tqdm(as_completed(futures), total=total_augmentations, desc="Applying augmentations"):
            pass
    
    # tries and returns error for every image that failed to be augmented - most likely bounding box boundary error
    for future in futures:
        if not future.done():
            print("A future did not complete.")  
        try:
            future.result()  # re-raise exception that occurred
        except Exception as e:
            print(f"An error occurred: {e}")