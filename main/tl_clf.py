"""Python file accompanying the tl_clf.ipynb notebook."""

# Import modules
import dataset as ds
import config as cfg
from evaluate import Evaluate
import tools
seed_const = 42

from sklearn.model_selection import train_test_split
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Rescaling, Dropout
from tensorflow.keras.metrics import Precision, Recall

from clearml import Task
import clearml
clearml.browser_login()

import argparse



def main(RUN, augmentation_metadata, task_name, sub_project, epochs, train_size, w1, w2, new_image_count):

    seed_time = tools.generate_seed()
    print("Seed: ", seed_time)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


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

    # always train on Palo Alto for Consitency
    train_df = train_df[(train_df['location'] == 'Palo Alto')]
    valid_df = valid_df[(valid_df['location'] == 'Palo Alto')]



    # =================================================================================================
    # TEST PARAMETERS
    dataset_name = f'weather_{w1}{w2}_{train_size}'
    dataset_dir = cfg.CLF_DATASET_DIR + f'/{dataset_name}'
    project_dir = f'{cfg.CLF_PROJECT_DIR}/{dataset_name}/'
    class_names = cfg.CLF_CLASS_NAMES

    zoom_factor = 1.5
    N = int(epochs / 2)

    #task_name = 'auto-test9'
    task_name = f'{task_name}-w{w1}{w2}-{epochs}-{train_size}-{RUN}'
    project_name= cfg.CLF_PROJECT_NAME + f'/{sub_project}'


    val_size = int(train_size * 0.25)

    # =================================================================================================


    # create new datasets

    # fitler by weather
    filtered_train_df = train_df[(train_df['weather'] == w1) | (train_df['weather'] == w2)]
    filtered_valid_df = valid_df[(valid_df['weather'] == w1) | (valid_df['weather'] == w2)]

    # Create a combined stratification key
    filtered_train_df['stratify_key'] = filtered_train_df['ac'] + '_' + filtered_train_df['weather'].astype(str)
    filtered_valid_df['stratify_key'] = filtered_valid_df['ac'] + '_' + filtered_valid_df['weather'].astype(str)


    # Now use this stratification key in train_test_split
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
    # create dataset directory from dataframes above
    ds.create_sub_dataset(dataset_dir, test_train_df, test_val_df, class_names)

    # correct single class labels to accomodate for multi-class classification
    ds.correct_dataset_labels(dataset_dir, test_train_df, test_val_df, class_names)

    # augment dataset
    ds.augment_dataset(dataset_dir, augmentation_metadata)
    
    # append new images to dataset train set (only need for final CLF tests)
    ds.append_new_train_images(dataset_dir, new_image_count, filtered_train_df, seed_time, class_names)
    
    # Pre-processing to AID classification (apply zoom factor to all images)
    ds.pre_process_dataset_for_classification(dataset_dir, zoom_factor)

    # create class folders within train and valid directories for keras format
    ds.reorganize_dataset_for_keras(dataset_dir)



    # load datasets using keras

    tf.keras.backend.clear_session()
    
    train_dir = os.path.join(dataset_dir,'images','train')
    train_aug_dir = os.path.join(dataset_dir,'images','train-aug')
    valid_dir = os.path.join(dataset_dir,'images','valid')

    batch_size = 32 
    img_height = 256
    img_width = 256

    train_data = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        seed=seed_time,
        image_size=(img_height, img_width),
        shuffle=True)

    train_aug_data = tf.keras.utils.image_dataset_from_directory(
        train_aug_dir,
        batch_size=batch_size,
        seed=seed_time,
        image_size=(img_height, img_width),
        shuffle=True)

    valid_data = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        batch_size=batch_size,
        seed=seed_time,
        image_size=(img_height, img_width),    
        shuffle=True)


    tf.keras.utils.set_random_seed(seed_const)


    # hyper-parameters
    hyper_params = {
        'dataset': dataset_name,
        'original_dataset_size': {'train_size': train_size, 'val_size': val_size},
        'train_size': tools.count_images(train_dir),
        'train_aug_size': tools.count_images(train_aug_dir),
        'valid_size': tools.count_images(valid_dir),
        'epochs': epochs, 
        'N': N,
        'zoom_factor': zoom_factor, 
        'batch_size': batch_size, 
        'img_height': img_height, 
        'img_width': img_width, 
        'class_names': class_names
        }

    # convert labels to one-hot encoding to allow for more metrics to be tracked
    def one_hot_enc(image, label):
        return image, tf.one_hot(label, len(class_names))

    train_data = train_data.map(one_hot_enc)
    train_aug_data = train_aug_data.map(one_hot_enc)
    valid_data = valid_data.map(one_hot_enc)


    # optimise datasets
    #AUTOTUNE = tf.data.AUTOTUNE
    #train_data = train_data.cache().shuffle(1000, seed=seed_time).prefetch(buffer_size=AUTOTUNE)
    #train_aug_data = train_aug_data.cache().shuffle(1000, seed=seed_time).prefetch(buffer_size=AUTOTUNE)
    #valid_data = valid_data.cache().prefetch(buffer_size=AUTOTUNE)


    # get list of validation labels (y_true) for evaluation
    validation_labels = []
    for images, labels in valid_data:
        validation_labels.append(labels.numpy())

    validation_labels = np.concatenate(validation_labels)


    # CNN for image classification
    def get_model(): 
        # number of classes
        num_classes = len(class_names) 
        model = Sequential([
            Input(shape=(img_height, img_width, 3)), # inputs shape, height, width, channels (RGB)
            Rescaling(1./255), # normalize pixel values
            Conv2D(16, 3, padding='same', activation='relu'), # 16 filters, 3x3 kernel, relu activation
            MaxPooling2D(), # performs 2D max pooling
            Conv2D(32, 3, padding='same', activation='relu'), # 32 filters, 3x3 kernel, relu activation
            MaxPooling2D(), # 2D max pooling
            Conv2D(64, 3, padding='same', activation='relu'), # 64 filters, 3x3 kernel, relu activation
            MaxPooling2D(), # 2D max pooling
            Dropout(0.2), # dropout layer to increase regularization and reduce overfitting
            Flatten(), # flattens output of previos layer to 1D
            Dense(128, activation='relu'),      
            Dense(num_classes, activation='softmax') 
        ])
        
        # compile model  
        model.compile(optimizer='adam', # use Adam optimizer
                        loss='categorical_crossentropy', # use categorical crossentropy loss for mutliclass classification
                        metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]) # trach accuracy, precision, recall (and loss)

        # print model summary
        #model.summary()
        
        return model

    ############################################################################################################

    # Train Pure

    # local logs directory
    logs_dir=cfg.CLF_PROJECT_DIR
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_dir)

    #connect to clearml
    task = Task.init(project_name=project_name, task_name=f"{task_name}-pure")
    logger = task.get_logger()

    # clearml hyperparameters
    task.connect(hyper_params)

    # ensure model from scratch and get model

    model = get_model()

    # train model
    print('Training model...')
    train_hst = model.fit(
        train_data, 
        epochs=hyper_params['epochs'], 
        validation_data=valid_data,
        callbacks=[tensorboard_callback]
        )

    # predict on validation set
    print('Predicting on validation set...')
    y_pred = model.predict(valid_data)

    # evaluate
    pure_eval = Evaluate(train_hst, validation_labels, y_pred, class_names, aug=False, sf=3)

    # send metrics to clearML
    pure_eval.log_metrics(task, logger, N, None, hyper_params)
    

    # close task
    print("done")


    ############################################################################################################

    # Train Augmented

    # local logs directory, callbacks allow for real time metrics in clearML
    logs_dir=cfg.CLF_PROJECT_DIR
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_dir)

    #connect to clearml
    task = Task.init(project_name=project_name, task_name=f"{task_name}-aug")
    logger = task.get_logger()


    # clearml hyperparameters
    task.connect(hyper_params)

    # ensure model from scratch and get model
    model_aug = get_model()

    # train model
    print('Training model...')
    train_aug_hst = model_aug.fit(
        train_aug_data, 
        epochs=hyper_params['epochs'], 
        validation_data=valid_data, 
        callbacks=[tensorboard_callback]
        )

    # predict on validation set
    print('Predicting on validation set...')
    y_pred_aug = model_aug.predict(valid_data)

    # evaluate
    aug_eval = Evaluate(train_aug_hst, validation_labels, y_pred_aug, class_names, aug=True, sf=3)

    # send metrics to clearML
    aug_eval.log_metrics(task, logger, N, augmentation_metadata, hyper_params) 
    
    # close task
    print("done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with a specific run ID.')
    parser.add_argument('run_number', type=int, help='The run ID for this training session.')
    args = parser.parse_args()
    
    main(args.run_number)
