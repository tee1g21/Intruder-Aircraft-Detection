"""Python file accompanying the tl_clf.ipynb notebook."""

# Import modules
import dataset as ds
import config as cfg
import tools
from evaluate import Evaluate

import torch
from ultralytics import settings
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import gc

from clearml import Task
import clearml
clearml.browser_login()

import argparse


def main(RUN, augmentation_metadata, task_name, sub_project, epochs, train_size, w1, w2):

    seed_time = tools.generate_seed()
    print("Seed: ", seed_time)

    # Ultralytics settings
    settings.reset()
    settings.update({'datasets_dir': cfg.YOLO_DATASET_DIR.replace('/','\\')})

    # GPU Utilisation
    print("GPU availble:",torch.cuda.is_available())
    device = torch.device("cuda")
    #print(torch.backends.cudnn.version())

    # Import dataset
    # Base paths for the images and labels
    train_images_path = f'{cfg.BASE_DATASET}/images/train'
    train_labels_path = f'{cfg.BASE_DATASET}/labels/train'
    val_images_path = f'{cfg.BASE_DATASET}/images/valid'
    val_labels_path = f'{cfg.BASE_DATASET}/labels/valid'

    # Base path for metadata
    metadata_path = f'{cfg.BASE_DATASET}/metadata.json'

    # Create the DataFrames for the train and validation sets
    train_df = ds.create_dataframe(train_images_path, train_labels_path, metadata_path)
    valid_df = ds.create_dataframe(val_images_path, val_labels_path, metadata_path)


    # =================================================================================================
    # TEST PARAMETERS
    dataset_name = f'weather_{w1}{w2}_{train_size}'
    dataset_dir = f'{cfg.YOLO_DATASET_DIR}/{dataset_name}'
    project_dir = f'{cfg.YOLO_PROJECT_DIR}/{dataset_name}/'
    class_names = cfg.YOLO_CLASS_NAMES

    # task specifc training parameters
    model_variant = "yolov8n"

    task_name = f'{task_name}-w{w1}{w2}-{epochs}-{train_size}-{RUN}'
    project_name= cfg.YOLO_PROJECT_NAME + f'/{sub_project}'

    val_size = int(train_size * 0.2)

    # =================================================================================================

    # always train on Palo Alto for Consitency
    train_df = train_df[(train_df['location'] == 'Palo Alto')]
    valid_df = valid_df[(valid_df['location'] == 'Palo Alto')]


    # fitler by weather
    filtered_train_df = train_df[(train_df['weather'] == w1) | (train_df['weather'] == w2)]
    filtered_valid_df = valid_df[(valid_df['weather'] == w1) | (valid_df['weather'] == w2)]

    # Create a combined stratification key
    filtered_train_df['stratify_key'] = filtered_train_df['ac'] + '_' + filtered_train_df['weather'].astype(str)
    filtered_valid_df['stratify_key'] = filtered_valid_df['ac'] + '_' + filtered_valid_df['weather'].astype(str)

    _, test_train_df = train_test_split(
        filtered_train_df,
        test_size=train_size,  # Number of items you want in your sample
        stratify=filtered_train_df['stratify_key'],  # Stratify based on the combined column
        random_state=seed_time  # Ensures reproducibility
    )

    _, test_val_df = train_test_split(
        filtered_valid_df,
        test_size=val_size,  # Number of items you want in your sample
        stratify=filtered_valid_df['stratify_key'],  # Stratify based on the combined column
        random_state=seed_time  # Ensures reproducibility
    )

    # create sub dataset
    ds.create_sub_dataset(dataset_dir, test_train_df, test_val_df, class_names)

    # correct dataset labels
    ds.correct_dataset_labels(dataset_dir, test_train_df, test_val_df, class_names)

    # augment dataset
    ds.augment_dataset(dataset_dir, augmentation_metadata)


    # TRAIN PURE
    ############################################################################################################
    
    # train on pure dataset
    print("Training on pure dataset ...")

    # dataset location
    dataset_path=f'{dataset_dir}\\{dataset_name}.yaml'
    project =  project_dir + 'pure' #save_dir # weight save path

    # Create ClearML task
    task_pure = Task.init(
        project_name=project_name,
        task_name=f"{task_name}-pure"
    )
    task_pure.set_parameter("model_variant", model_variant)

    # Define Yolo model
    #model = YOLO(f'{model_variant}.yaml') # train on model which is not pre-trained
    model_pure = YOLO(f'{model_variant}.pt')

    #train args
    args_pure = dict(data=dataset_path, 
                epochs=epochs, 
                device=0, 
                #close_mosaic=epochs, # disable mosaic augmentation
                seed=42
                )
    task_pure.connect(args_pure)

    # train model
    results_pure=model_pure.train(**args_pure, project=project)

    # validate model
    metrics_pure = model_pure.val()
    metrics_dict_pure = Evaluate.get_yolo_metrics_dict(metrics_pure)

    # Log the metrics to ClearML
    print("Uploading metrics to ClearML ...")
    task_pure.upload_artifact('VAL_METRICS', tools.pretty_print_dict(metrics_dict_pure))

    print("Closing Task Pure ...")
    task_pure.close()
    print("done")

    #model_pure = None
    #task_pure = None
    #args_pure = None
    #results_pure = None
    #metrics_pure = None
    #gc.collect()

    ############################################################################################################

    # Train Augmented
    print("Training on augmented dataset ...")
    
    # dataset location
    dataset_path=f'{dataset_dir}\\{dataset_name}-aug.yaml'
    project =  project_dir + 'augmented' #save_dir # weight save path

    # Create ClearML task
    task_aug = Task.init(
        project_name=project_name,
        task_name=f"{task_name}-aug"
    )
    task_aug.set_parameter("model_variant", model_variant) 

    # Define Yolo model
    #model = YOLO('yolov8n.yaml')
    model_aug = YOLO('yolov8n.pt')

    # train model
    results_aug = model_aug.train(data=dataset_path, 
                        epochs=epochs, 
                        device=0, 
                        project=project,
                        #close_mosaic=epochs, # disable mosaic augmentation
                        seed=42)

    # validate model
    metrics_aug = model_aug.val()
    metrics_dict_aug = Evaluate.get_yolo_metrics_dict(metrics_aug)

    # Log the metrics to ClearML
    print("Uploading metrics to ClearML ...")
    task_aug.upload_artifact('VAL_METRICS', tools.pretty_print_dict(metrics_dict_aug))
    task_aug.upload_artifact('AUGMENTATION_METADATA', tools.pretty_print_dict(augmentation_metadata))

    print("Closing Task Augmented ...")
    task_aug.close()

    print("done")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with a specific run ID.')
    parser.add_argument('run_number', type=int, help='The run ID for this training session.')
    args = parser.parse_args()
    
    main(args.run_number)
