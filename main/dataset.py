"""
This dataset module provides comprehensive tools for managing, augmenting, and organizing datasets. It includes functions to create data frames,
copy files concurrently, augment datasets, and reorganize them for compatibility with frameworks like Keras.

Key Functions:
- create_dataframe: Creates a DataFrame from image and label paths, integrating metadata for comprehensive dataset management.
- copy_dataframe_files_concurrently: Copies image and label files listed in a DataFrame to specified directories using concurrent execution.
- create_sub_dataset: Establishes a structured sub-dataset within a specified directory for targeted analysis or processing.
- augment_dataset: Augments a dataset by applying specified methods, enhancing data variability and robustness for training models.
- reorganize_dataset_for_keras: Organizes a dataset into class-specific subdirectories suitable for Keras' data loading utilities.
- append_new_train_images: Expands an existing dataset with additional training images to ensure diverse and balanced data.

"""

import tools
import augmenter

import os
import pandas as pd
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
max_workers=12
from tqdm import tqdm
import yaml
import random
import cv2
from sklearn.model_selection import train_test_split


# Function to create a DataFrame from images and labels
def create_dataframe(images_path, labels_path, metadata_path):
    """
    Loads image and label paths and their corresponding metadata into a DataFrame.

    Parameters:
        images_path (str): Path to the directory of image files.
        labels_path (str): Path to the directory of label files.
        metadata_path (str): Path to the metadata JSON file.

    Returns:
        DataFrame: DataFrame combining image and label paths with expanded metadata.
    """

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
    """
    Copies image and label files listed in a DataFrame to specified directories using concurrent execution.

    Parameters:
        df (DataFrame): Contains columns 'image_path' and 'label name'.
        img_dest_dir (str): Destination directory for copied image files.
        label_dest_dir (str): Destination directory for copied label files.

    The function uses a ThreadPoolExecutor to handle multiple copy operations simultaneously, 
    enhancing efficiency especially for large datasets.
    """
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
    """
    Sets up a sub-dataset within a specified directory by organizing and copying training and validation files 
    according to the provided dataframes. This function also creates a YAML configuration for the sub-dataset.

    Parameters:
        dataset_dir (str): Base directory where the sub-dataset will be created.
        filtered_train_df (DataFrame): DataFrame with paths to training images and labels.
        filtered_valid_df (DataFrame): DataFrame with paths to validation images and labels.
        class_names (list): List of class names corresponding to the dataset labels.

    The function first clears any pre-existing data in the dataset directory, sets up the required directory structure,
    copies the necessary files, and then writes a YAML configuration file reflecting the dataset structure.
    """    
    
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
def pre_process_dataset_for_classification(dataset_dir, zoom_factor):
    """
    Processes image datasets for classification by applying a zoom operation to each image 
    and handles the dataset organization for training, augmented training, and validation sets.

    Parameters:
        dataset_dir (str): The base directory of the dataset where images and labels are stored.
        zoom_factor (float): The factor by which images will be zoomed.

    This function iterates over specified subdirectories ('train', 'train-aug', 'valid'), 
    applies a zoom augmentation to each image, and updates the dataset by saving the processed images and labels.
    It uses concurrent processing to speed up the task and handles any exceptions that may occur during the process.
    """

    for set_type in ['train', 'train-aug', 'valid']:
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


# Function to update the labels in a given directory
def update_labels(dataset_dir, labels_path, label_mapping, type):
    """
    Updates the class indices in label files based on a provided mapping, for a specified dataset type.

    Parameters:
        dataset_dir (str): Directory containing the dataset.
        labels_path (str): Path to the directory containing the label files.
        label_mapping (dict): Dictionary mapping label filenames to new class indices.
        type (str): The subset of the dataset being processed (e.g., 'train', 'valid').

    This function iterates through each label file specified in the label mapping. 
    If the label file exists, it reads and updates the class indices according to the mapping,
    and then rewrites the modified labels back to the file.
    """

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

# corrects default YOLO labels to match class names for sub-dataset
def correct_dataset_labels(dataset_dir, train_df, val_df, class_names):
    """
    Corrects the class indices in the dataset's label files for both training and validation sets according to specified class names.

    Parameters:
        dataset_dir (str): The root directory of the dataset containing label subdirectories.
        train_df (DataFrame): DataFrame containing paths to training label files and their corresponding class annotations.
        val_df (DataFrame): DataFrame containing paths to validation label files and their corresponding class annotations.
        class_names (list of str): List of class names used to index class annotations.

    This function maps the filenames to new class indices derived from the class names. It then updates the label files in 
    both the training and validation directories to reflect these new indices.
    """

    
    # Assuming dataset_dir is the root that contains 'labels/train' and 'labels/valid'
    train_labels_path = dataset_dir + f'/labels/train'
    val_labels_path = dataset_dir + f'/labels/valid'
    
    # Creating dictionaries to map filenames to new class indices based on class_names
    train_label_mapping = {os.path.basename(row['label_path']): class_names.index(row['ac']) for _, row in train_df.iterrows()}
    val_label_mapping = {os.path.basename(row['label_path']): class_names.index(row['ac']) for _, row in val_df.iterrows()}

    # Update labels in both train and validation directories using respective mappings
    update_labels(dataset_dir, train_labels_path, train_label_mapping, 'train')
    update_labels(dataset_dir, val_labels_path, val_label_mapping, 'valid')

    print("Label correction completed.")

# creates augmented dataset structure
def create_augmented_dataset_structure(original_dataset_path):    
    """
    Creates an augmented dataset structure by adding 'train-aug' directories for images and labels 
    and updating the YAML configuration to point to these new directories.

    Parameters:
        original_dataset_path (str): The path to the root of the original dataset.

    This function generates additional directories for storing augmented training data and 
    creates a new YAML file to manage the dataset configuration. It ensures any existing augmented 
    directories or configuration files are removed before creating new ones.
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
    """
    Copies all files from one directory to another using concurrent execution to speed up the process.

    Parameters:
        src_dir (str): The directory from which to copy files.
        dst_dir (str): The target directory where files will be copied.

    This function uses a ThreadPoolExecutor to perform the copying tasks concurrently, 
    enhancing efficiency especially when dealing with large numbers of files.
    """
    # Retrieve a list of source file paths
    src_files = [os.path.join(src_dir, file_name) for file_name in os.listdir(src_dir)]
    dst_files = [os.path.join(dst_dir, os.path.basename(file_path)) for file_path in src_files]
    
    # Copy files concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(tools.copy_file, src_files, dst_files), total=len(src_files), desc="Copying files"))

# Creates augmented dataset, using metadata dictionary to define augmentation methods
def augment_dataset(original_dataset_path, augmentation_metadata):
    """
    Augments a dataset by copying the original files to new directories and applying specified augmentation methods.

    Parameters:
        original_dataset_path (str): Path to the root directory of the dataset.
        augmentation_metadata (dict): Contains information about the augmentation methods to apply,
                                    including method names and parameters.

    This function first creates augmented dataset structures, then copies existing dataset files into these new directories.
    It applies the augmentations defined in the metadata to a subset of the images, using concurrent processing for efficiency.
    Each augmentation method can potentially modify both the images and their corresponding labels.
    """
    
    # Reconstruct dataset with augmentation directories
    create_augmented_dataset_structure(original_dataset_path)

    # Define paths to directories
    images_dir = os.path.join(original_dataset_path, 'images/train/')
    images_aug_dir = os.path.join(original_dataset_path, 'images/train-aug/')
    labels_dir = os.path.join(original_dataset_path, 'labels/train/')
    labels_aug_dir = os.path.join(original_dataset_path, 'labels/train-aug/')

    # Copy train image and label contents to augmented directories
    copy_directory_contents_concurrently(images_dir, images_aug_dir)
    copy_directory_contents_concurrently(labels_dir, labels_aug_dir)

    # List of training images in augmented directory
    image_paths = [os.path.join(images_aug_dir, img_name) for img_name in os.listdir(images_aug_dir)]

    # Apply augmentations based on metadata
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for method_name, method_info in augmentation_metadata['methods'].items():
            percentage = method_info['apply_to_percentage']
            num_images_to_augment = int(len(image_paths) * percentage)
            sampled_images = random.choices(image_paths, k=num_images_to_augment)  # Allows repetition

            for image_path in sampled_images:
                label_path = image_path.replace(images_aug_dir, labels_aug_dir).replace('.jpg', '.txt')
                futures.append(executor.submit(augmenter.augment_image, image_path, images_aug_dir, label_path, labels_aug_dir, method_name, method_info))

        # Progress bar for the augmentation tasks
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Applying augmentations"):
            pass

    # Check for errors in futures
    for future in futures:
        if not future.done():
            print("A future did not complete.")
        try:
            future.result()  # Re-raise exception that occurred
        except Exception as e:
            print(f"An error occurred: {e}")


# reorganizes dataset for keras         
def reorganize_dataset_for_keras(dataset_dir):
    """
    Reorganizes a dataset into a structure compatible with Keras by sorting images into class-specific subdirectories.

    Parameters:
        dataset_dir (str): The root directory of the dataset which contains the images and labels.

    This function loads class names from a YAML file, creates class-specific subdirectories under each image directory,
    and moves images into these subdirectories based on their class ID obtained from the corresponding label files.
    Designed to facilitate image classification tasks using Keras, this organization supports direct usage of Keras's
    data loading utilities that expect data in such a structured format.
    """
    
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
    

# adds images to train set to make train and train-aug equal in size
def append_new_train_images(dataset_dir, N, master_df, seed_time, class_names):
    """
    Appends a specified number of new training images and labels to an existing dataset directory from a master DataFrame.

    Parameters:
        dataset_dir (str): Path to the dataset directory.
        N (int): Number of new images to add to the training set.
        master_df (DataFrame): A DataFrame containing paths and metadata for potential training images.
        seed_time (int): Seed for random operations to ensure reproducibility.
        class_names (list of str): List of class names used to update labels based on classification.

    This function filters out images already present in the dataset, samples new ones based on specified requirements,
    and copies them to the dataset's training directories. It also updates the class indices in label files based on
    newly added images. This is intended to augment the training data while maintaining class balance and diversity.
    """    
    
    # image and label directories
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / 'images' / 'train'
    labels_dir = dataset_dir / 'labels' / 'train'

    # list the existing images in the train directory
    existing_images = set([file.name for file in images_dir.glob('*.jpg')])

    # Filter parent dataframe to only include entries not already in the dataset
    available_df = master_df[~master_df['image_path'].apply(lambda x: Path(x).name in existing_images)]

    # If the stratify column is not provided in the DataFrame, it will be generated here.
    if 'stratify_key' not in available_df.columns:
        available_df['stratify_key'] = available_df['ac'] + '_' + available_df['weather'].astype(str)

    # Ensure that there are enough images to sample from
    if len(available_df) < N:
        raise ValueError("Not enough unique images available to meet the request.")

    # Split to get the required number of unique new images
    _, new_train_df = train_test_split(
        available_df,
        test_size=N,  # Number of items you want in your sample
        stratify=available_df['stratify_key'],  # Stratify based on the combined column
        random_state=seed_time  # Ensures reproducibility
    )

    # Prepare files to be copied using multithreading
    tasks = []
    for _, row in new_train_df.iterrows():
        source_image_path = Path(row['image_path'])
        source_label_path = Path(row['label_path'])

        target_image_path = images_dir / source_image_path.name
        target_label_path = labels_dir / source_label_path.name

        tasks.append((source_image_path, target_image_path))
        tasks.append((source_label_path, target_label_path))

    # Use ThreadPoolExecutor to copy files concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(lambda x: tools.copy_file(*x), tasks), total=len(tasks), desc="Appending train files"))

    train_label_mapping = {os.path.basename(row['label_path']): class_names.index(row['ac']) for _, row in new_train_df.iterrows()}
    update_labels(dataset_dir, labels_dir, train_label_mapping, 'train')
