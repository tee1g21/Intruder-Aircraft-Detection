"""Python File containing methods to handle the main dataset, create sub datasets and to augment these datasets"""

import tools
import augmenter

import os
import pandas as pd
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
max_workers=12
from tqdm.auto import tqdm
import yaml
import random
import cv2

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
            futures = [executor.submit(tools.copy_file, row['image_path'], img_dest_dir / f"{Path(row['image_path']).name}") for _, row in df.iterrows()]
            futures += [executor.submit(tools.copy_file, row['label_path'], label_dest_dir / f"{Path(row['label_path']).name}") for _, row in df.iterrows()]
            
            # Initialize progress bar
            pbar = tqdm(total=len(futures), desc='Copying files')
            for future in as_completed(futures):
                # Update progress bar upon task completion
                pbar.update(1)
            pbar.close()

# creates subsets of main dataset
def create_sub_dataset(dataset_dir, filtered_train_df, filtered_valid_df, class_names):
    
    dataset_dir = Path(dataset_dir)
    dataset_name = os.path.basename(dataset_dir)
    images_dir = Path(dataset_dir) / 'images'
    labels_dir = Path(dataset_dir) / 'labels'

    print("Removing dataset if pre-existing")
    tools.remove_if_exists(dataset_dir)

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
        'path': str(f'{dataset_dir}').replace('\\', '/'),  # Ensuring forward slashes
        'train': str(f'{images_dir}/train').replace('\\', '/'),
        'val': str(f'{images_dir}/valid').replace('\\', '/'),
        'names': {index: name for index, name in enumerate(class_names)}
    }  

    yaml_path = dataset_dir / f"{dataset_name}.yaml"
    with open(yaml_path, 'w') as file:
        yaml.dump(yaml_content, file, sort_keys=False)

    print(f"Dataset '{dataset_name}' created at {dataset_dir}")

# zoom into aircraft on every dataset image, including validation, so to so that the aircraft is classified
def pre_process_dataset_for_classification(dataset_dir, zoom_factor=2, max_workers=4):
    for set_type in ['train', 'valid']:
        image_dir = os.path.join(dataset_dir, 'images', set_type)
        label_dir = os.path.join(dataset_dir, 'labels', set_type)
        
        # Get all image file paths
        image_paths = [os.path.join(image_dir, image_name) for image_name in os.listdir(image_dir) if image_name.endswith('.jpg')]
        
        # Process each image
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(augmenter.zoom, image_path, label_path, zoom_factor): (image_path, label_path)
                       for image_path in image_paths
                       for label_path in [os.path.join(label_dir, os.path.splitext(os.path.basename(image_path))[0] + '.txt')]}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
                image_path, label_path = futures[future]
                try:
                    zoomed_image, zoomed_label = future.result()  # This will re-raise any exceptions that occurred
                    # Save the zoomed image and updated label
                    tools.save_image(image_path, zoomed_image)
                    tools.save_label(label_path, zoomed_label)
                except Exception as e:
                    print(f"An error occurred with {image_path}: {e}")
                    # If error occurred during processing, delete the original image and label
                    os.remove(image_path)
                    os.remove(label_path)

# corrects default YOLO labels to match class names for sub-dataset
def correct_dataset_labels(dataset_dir, train_df, val_df, class_names):
    # Assuming dataset_dir is the root that contains 'labels/train' and 'labels/valid'
    train_labels_path = dataset_dir + f'/labels/train'
    val_labels_path = dataset_dir + f'/labels/valid'
    
    # Creating dictionaries to map filenames to new class indices based on class_names
    train_label_mapping = {os.path.basename(row['label_path']): class_names.index(row['ac']) for _, row in train_df.iterrows()}
    val_label_mapping = {os.path.basename(row['label_path']): class_names.index(row['ac']) for _, row in val_df.iterrows()}

    # Function to update the labels in a given directory
    def update_labels(labels_path, label_mapping, type):
        print(f'Processing {type} labels in {os.path.basename(dataset_dir)}:')
        for label_filename in tqdm(label_mapping, desc=f'Processing labels'):
            new_class_index = label_mapping[label_filename]
            label_file_path = os.path.join(labels_path, label_filename)
            if os.path.isfile(label_file_path):
                with open(label_file_path, 'r') as file:
                    lines = file.readlines()

                updated_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        parts[0] = str(new_class_index)  # Update the class index
                        updated_lines.append(' '.join(parts))
                
                with open(label_file_path, 'w') as file:
                    file.writelines('\n'.join(updated_lines))
            else:
                print(f"File not found: {label_file_path}")

    # Update labels in both train and validation directories using respective mappings
    update_labels(train_labels_path, train_label_mapping, 'train')
    update_labels(val_labels_path, val_label_mapping, 'valid')

    print("Label correction completed.")

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
            tools.remove_if_exists(path, progress)

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
        list(tqdm(executor.map(tools.copy_file, src_files, dst_files), total=len(src_files), desc="Copying files"))

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
                futures.append(executor.submit(augmenter.augment_image, image_path, images_aug_dir, label_path, labels_aug_dir, method_name, method_info))

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
            
def reorganize_dataset_for_keras(dataset_dir):
    
    # extract dataset name
    dataset_name = os.path.basename(dataset_dir)
    
    # Load class names from the base YAML file
    base_yaml_path = os.path.join(dataset_dir, f'{dataset_name}.yaml')
    with open(base_yaml_path) as f:
        base_data = yaml.load(f, Loader=yaml.FullLoader)
    class_names = base_data['names']

    # Function to create class subdirectories and move images
    def organize_images(base_path, label_rel_path):
        for class_id, class_name in class_names.items():
            # Create class subdirectories within the image directories
            class_image_dir = os.path.join(base_path, class_name)
            os.makedirs(class_image_dir, exist_ok=True)

        # Get list of label files for progress bar
        label_dir = os.path.join(dataset_dir, label_rel_path)
        label_files = os.listdir(label_dir)

        # Wrap label_files with tqdm for a progress bar
        for label_file in tqdm(label_files, desc=f'Moving images in {os.path.basename(base_path)}'):
            # Read the label to get the class ID
            with open(os.path.join(label_dir, label_file), 'r') as file:
                class_id = int(file.readline().split()[0])

            class_name = class_names[class_id]
            image_file = label_file.replace('.txt', '.jpg')
            source_image_path = os.path.join(base_path, image_file)
            dest_image_path = os.path.join(base_path, class_name, image_file)

            if os.path.isfile(source_image_path):
                tools.move_file(source_image_path, dest_image_path)

    # Organize the images in train, train-aug, and val directories
    organize_images(os.path.join(dataset_dir, 'images', 'train'), 'labels/train')
    organize_images(os.path.join(dataset_dir, 'images', 'train-aug'), 'labels/train-aug')
    organize_images(os.path.join(dataset_dir, 'images', 'valid'), 'labels/valid')