"""Python file accompanying the tl_clf.ipynb notebook."""

# Import modules
import dataset as ds
import config as cfg
import tools

import torch
from ultralytics import settings
from ultralytics import YOLO
from sklearn.model_selection import train_test_split

from clearml import Task
import clearml
clearml.browser_login()
import gc
import argparse


def main(RUN):

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
    dataset_name = 'test500'

    # project and task constant parameters
    #project_name= cfg.YOLO_PROJECT_NAME
    dataset_dir = f'{cfg.YOLO_DATASET_DIR}/{dataset_name}'
    project_dir = f'{cfg.YOLO_PROJECT_DIR}/{dataset_name}/'
    class_names = cfg.YOLO_CLASS_NAMES

    # task specifc training parameters
    epochs = 5
    model_variant = "yolov8n"

    task_name= task_name = 'epoch_test'
    task_name = f'{task_name}-{epochs}-{RUN}'
    project_name= cfg.YOLO_PROJECT_NAME + f'/epoch-test'

    """
    Methods: 
    - flip
    - rotate
    - bnc
    - gaussian
    - histEq
    - whiteBal
    - sharpen
    - zoom
    - hsv
    """

    augmentation_metadata = {
        'methods': {        
            'flip': {
                'parameters': {
                    'orientation': 'h',  # Could be 'h' for horizontal or 'v' for vertical
                    'p': 1.0  # Probability of applying the augmentation
                },
                'apply_to_percentage': 0.5  # 50% of the training images
            }        
        }
    }

    train_size = 250
    val_size = int(train_size * 0.2)

    # =================================================================================================


    # create new datasets

    _, test_train_df = train_test_split(
    train_df,
    test_size=train_size,  # Number of items you want in your sample
    stratify=train_df['ac'],  # Stratify based on the combined column
    random_state=seed_time  # Ensures reproducibility
    )   

    _, test_val_df = train_test_split(
        valid_df,
        test_size=val_size,  # Number of items you want in your sample
        stratify=valid_df['ac'],  # Stratify based on the combined column
        random_state=seed_time  # Ensures reproducibility
    )

    # create sub dataset
    ds.create_sub_dataset(dataset_dir, test_train_df, test_val_df, class_names)

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
    task = Task.init(
        project_name=project_name,
        task_name=f"{task_name}-pure"
    )
    task.set_parameter("model_variant", model_variant)

    # Define Yolo model
    #model = YOLO(f'{model_variant}.yaml') # train on model which is not pre-trained
    model = YOLO(f'{model_variant}.pt')

    #train args
    args = dict(data=dataset_path, 
                epochs=epochs, 
                device=0, 
                #close_mosaic=epochs, # disable mosaic augmentation
                seed=42
                )
    task.connect(args)

    # train model
    results =model.train(**args, project=project)

    # validate model
    metrics = model.val()
    metrics_dict = {
        'mAP_50-95': metrics.box.map,     # Mean Average Precision from IoU=0.50 to 0.95
        'mAP_50': metrics.box.map50,      # Mean Average Precision at IoU=0.50
        'mAP_75': metrics.box.map75,      # Mean Average Precision at IoU=0.75
        'mAP_per_class': metrics.box.maps, # List of mAP from IoU=0.50 to 0.95 for each category
        'Precision': metrics.box.mp,      # Precision
        'Recall': metrics.box.mr          # Recall
    }

    print("Uploading metrics to ClearML ...")
    task.upload_artifact('VAL_METRICS', tools.pretty_print_dict(metrics_dict))

    # close task for next run
    task.close()
    print("done")

    #clear variables from memory
    task = None
    model = None
    args = None
    results = None


    ############################################################################################################

    # Train Augmented
    print("Training on augmented dataset ...")
    
    # clear variables
    torch.cuda.empty_cache()
    gc.collect()


    # dataset location
    dataset_path=f'{dataset_dir}\\{dataset_name}-aug.yaml'
    project =  project_dir + 'augmented' #save_dir # weight save path

    # Create ClearML task
    task = Task.init(
        project_name=project_name,
        task_name=f"{task_name}-aug"
    )
    task.set_parameter("model_variant", model_variant) 

    # Define Yolo model
    #model = YOLO('yolov8n.yaml')
    model = YOLO('yolov8n.pt')

    # train model
    args = dict(data=dataset_path, 
                epochs=epochs, 
                device=0, 
                #close_mosaic=epochs, # disable mosaic augmentation
                seed=42
                )
    task.connect(args)

    # train model
    results =model.train(**args, project=project)

    # validate model
    metrics = model.val()
    metrics_dict = {
        'mAP_50-95': metrics.box.map,     # Mean Average Precision from IoU=0.50 to 0.95
        'mAP_50': metrics.box.map50,      # Mean Average Precision at IoU=0.50
        'mAP_75': metrics.box.map75,      # Mean Average Precision at IoU=0.75
        'mAP_per_class': metrics.box.maps, # List of mAP from IoU=0.50 to 0.95 for each category
        'Precision': metrics.box.mp,      # Precision
        'Recall': metrics.box.mr          # Recall
    }

    print("Uploading metrics to ClearML ...")
    task.upload_artifact('VAL_METRICS', tools.pretty_print_dict(metrics_dict))
    task.upload_artifact('AUGMENTATION_METADATA', tools.pretty_print_dict(augmentation_metadata))


    # close task for next/last run
    task.close()
    print("done")
    
    model = None
    args = None
    results = None
    metrics = None
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with a specific run ID.')
    parser.add_argument('run_number', type=int, help='The run ID for this training session.')
    args = parser.parse_args()
    
    main(args.run_number)
