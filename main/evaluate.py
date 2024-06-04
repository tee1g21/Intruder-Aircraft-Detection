import tools

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import io 
from PIL import Image
import json

class Evaluate():
    """
    Evaluation class for analyzing and reporting the performance of machine learning models.

    This class provides methods to calculate, log, and visualize various performance metrics for models,
    including accuracy, precision, recall, F1 score, loss, AUC, and confusion matrix. It is designed to work
    with training history and prediction results, offering a comprehensive set of tools for model evaluation.

    Methods:
        __init__(self, train_hst, y_true, y_pred, class_names, aug=False, sf=3):
            Initializes the Evaluation object with training history, true and predicted labels, class names,
            augmentation type, and significant figures for rounding metrics.
        
        average_accuracy(self, print_bool=False):
            Calculates and returns the average training and validation accuracy.
        
        std_accuracy(self, print_bool=False):
            Calculates and returns the standard deviation of training and validation accuracy.
        
        best_accuracy(self, print_bool=False):
            Calculates and returns the highest training and validation accuracy.
        
        last_accuracy(self, print_bool=False):
            Retrieves and returns the accuracy from the final training epoch for both training and validation sets.
        
        overall_auc(self, print_bool=False):
            Calculates and returns the overall AUC for the ROC curve.
        
        max_f1(self, print_bool=False):
            Calculates and returns the maximum F1 score and the corresponding epoch.
        
        min_loss(self, print_bool=False):
            Calculates and returns the minimum validation loss and the corresponding epoch.
        
        diff_avg_loss_lastN(self, N, print_bool=False):
            Calculates and returns the difference between average validation and training loss over the last N epochs.
        
        std_loss_lastN(self, N, print_bool=False):
            Calculates and returns the standard deviation of the validation loss over the last N epochs.
        
        confusion_matrix_class_report(self, print_bool=False):
            Generates and returns a confusion matrix as a PIL image, a dictionary, and a classification report.
        
        class_auc(self, print_bool=False):
            Calculates and returns the AUC for each class.
        
        log_metrics(self, task, logger, N, augmentation_metadata, hyperparameters):
            Logs various performance metrics to a logging service and registers them as artifacts for tracking and analysis.
        
        get_yolo_metrics_dict(metrics):
            Converts YOLO evaluation metrics into a dictionary with specific key-value pairs.
    """   
    
    # initialize the class with train_hst, y_true, y_pred, class_names, type, print = False
    def __init__(self, train_hst, y_true, y_pred, class_names, aug=False, sf=3):   
        """Initializes an Evaluation object that captures and processes performance metrics for a trained model.

        Parameters:
            train_hst (History): A History object from Keras, containing training and validation metrics across epochs.
            y_true (list): True class labels for the test dataset.
            y_pred (list): Predicted class labels from the model.
            class_names (list of str): Names of the classes corresponding to indices in `y_true` and `y_pred`.
            aug (bool, optional): Indicates whether the evaluation is for an augmented dataset (default is False).
            sf (int, optional): Number of significant figures to use when displaying metrics (default is 3).

        Attributes:
            accuracy (float): Training accuracy.
            val_accuracy (float): Validation accuracy.
            precision (float): Training precision.
            val_precision (float): Validation precision.
            recall (float): Training recall.
            val_recall (float): Validation recall.
            loss (float): Training loss.
            val_loss (float): Validation loss.
            y_true (list): Actual labels.
            y_pred (list): Predicted labels.
            class_names (list of str): Class names.
            type (str): 'augmented' or 'pure', based on the dataset used.
            sf (int): Significant figures for formatting.
        """
        
        # accuracy
        self.accuracy = train_hst.history['accuracy']
        self.val_accuracy = train_hst.history['val_accuracy']
        
        # precision
        self.precision = train_hst.history['precision']
        self.val_precision = train_hst.history['val_precision']
        
        # recall
        self.recall = train_hst.history['recall']
        self.val_recall = train_hst.history['val_recall']
        
        # loss
        self.loss = train_hst.history['loss']
        self.val_loss = train_hst.history['val_loss']
        
        # y_true, y_pred
        self.y_true = y_true
        self.y_pred = y_pred
        
        # string class names
        self.class_names = class_names
        
        # define test type for printing       
        if aug == True:
            self.type = 'augmented'
        else:
            self.type = 'pure'    
               
        # significant figures
        self.sf = sf
        
    # return average accuracy: train, val
    def average_accuracy(self, print_bool = False):
        """Calculates the average training and validation accuracy over all epochs.

        Parameters:
            print_bool (bool, optional): If True, prints the average accuracies; default is False.

        Returns:
            tuple: A tuple containing two floats:
                1. Average training accuracy.
                2. Average validation accuracy.
        """
        
        average = round(np.mean(self.accuracy), self.sf)
        average_val = round(np.mean(self.val_accuracy), self.sf)
        
        if print_bool == True:
            print(f"Average Accuracy - {self.type}: {average}")
            print(f"Average Val Accuracy - {self.type}: {average_val}")
            
        return average, average_val
    
    # return standard deviation accuracy: train, val
    def std_accuracy(self, print_bool = False):
        """
        Calculates the standard deviation of training and validation accuracy over all epochs.

        Parameters:
            print_bool (bool, optional): If True, prints the standard deviations; default is False.

        Returns:
            tuple: A tuple containing two floats:
                1. Standard deviation of training accuracy.
                2. Standard deviation of validation accuracy.

        This method computes the standard deviations to measure the variability of accuracy scores
        throughout the training process, optionally printing them for 'pure' or 'augmented' evaluation types.
        """

        std = round(np.std(self.accuracy), self.sf)
        std_val = round(np.std(self.val_accuracy), self.sf)
        
        if print_bool == True:
            print(f"Std Accuracy - {self.type}: {std}")
            print(f"Std Val Accuracy - {self.type}: {std_val}")
            
        return std, std_val
    
    # return best epoch accuracy: train, val
    def best_accuracy(self, print_bool = False):
        """
        Calculates the highest training and validation accuracy achieved over all epochs.

        Parameters:
            print_bool (bool, optional): If True, prints the best accuracies; default is False.

        Returns:
            tuple: A tuple containing two floats:
                1. Best training accuracy.
                2. Best validation accuracy.

        This method finds the maximum accuracy values from the training history,
        optionally printing them based on the evaluation type (either 'pure' or 'augmented').
        """

        best = round(np.max(self.accuracy), self.sf)
        best_val = round(np.max(self.val_accuracy), self.sf)
        
        if print_bool == True:
            print(f"Best Accuracy - {self.type}: {best}")
            print(f"Best Val Accuracy - {self.type}: {best_val}")
            
        return best, best_val
    
    # return last epoch accuracy: train, val
    def last_accuracy(self, print_bool = False):
        """
        Retrieves the training and validation accuracy from the final epoch.

        Parameters:
            print_bool (bool, optional): If True, prints the final epoch accuracies; default is False.

        Returns:
            tuple: A tuple containing two floats:
                1. Training accuracy from the final epoch.
                2. Validation accuracy from the final epoch.

        This method accesses the last recorded accuracy values from the training history,
        optionally printing them based on the evaluation type (either 'pure' or 'augmented').
        """
        
        last = round(self.accuracy[-1], self.sf)
        last_val = round(self.val_accuracy[-1], self.sf)
        
        if print_bool == True:
            print(f"Final Epoch Accuracy - {self.type}: {last}")
            print(f"Final Epoch Val Accuracy - {self.type}: {last_val}")
            
        return last, last_val
    
    # overall area under the ROC curve from validation data
    def overall_auc(self, print_bool = False):
        """
        Calculates the overall Area Under the Curve (AUC) for the receiver operating characteristic (ROC).

        Parameters:
            print_bool (bool, optional): If True, prints the overall AUC; default is False.

        Returns:
            float: The overall AUC value.

        This method computes the AUC score based on the true labels and predicted probabilities,
        optionally printing the result based on the evaluation type (either 'pure' or 'augmented').
        """
        
        auc = round(roc_auc_score(self.y_true, self.y_pred), self.sf)
        
        if print_bool == True:
            print(f"Overall AUC - {self.type}: {auc}")
            
        return auc
    
    # max F1 score from validation data: max_f1_score, max_f1_epoch
    def max_f1(self, print_bool = False):
        """
        Calculates the maximum F1 score from the validation precision and recall values over all epochs.

        Parameters:
            print_bool (bool, optional): If True, prints the maximum F1 score and its corresponding epoch; default is False.

        Returns:
            tuple: A tuple containing:
                1. The maximum F1 score (float).
                2. The epoch at which the maximum F1 score was achieved (int).

        This method computes the F1 score for each epoch using validation precision and recall,
        identifies the highest F1 score and the epoch at which it occurs, and optionally prints these values.
        """

        f1_scores = []
        for p, r in zip(self.val_precision, self.val_recall):
            if p + r > 0:  
                f1 = 2 * p * r / (p + r)
                f1_scores.append(f1)
            else:
                f1_scores.append(0)

        # Get the maximum F1 score and its corresponding epoch
        max_f1_score = max(f1_scores)
        max_f1_epoch = f1_scores.index(max_f1_score)
        
        max_f1_score = round(max_f1_score, self.sf)
        
        if print_bool == True:
            print(f"Max F1 score - {self.type}: ", max_f1_score, " at epoch: ", max_f1_epoch)
            
        return max_f1_score, max_f1_epoch
    
    # min loss from validation data: min_loss, min_loss_epoch
    def min_loss(self, print_bool = False):
        """
        Calculates the minimum validation loss over all epochs and identifies the corresponding epoch.

        Parameters:
            print_bool (bool, optional): If True, prints the minimum validation loss and its corresponding epoch; default is False.

        Returns:
            tuple: A tuple containing:
                1. The minimum validation loss (float).
                2. The epoch at which the minimum validation loss was achieved (int).

        This method finds the lowest validation loss from the training history, identifies the epoch at which it occurred,
        and optionally prints these values based on the evaluation type (either 'pure' or 'augmented').
        """

        min_loss = np.min(self.val_loss)
        min_loss_epoch = round(self.val_loss.index(min_loss), self.sf)
        
        min_loss = round(min_loss, self.sf)
        
        if print_bool == True:
            print(f"Min Loss - {self.type}: ", min_loss, " at epoch: ", min_loss_epoch)
            
        return min_loss, min_loss_epoch
    
    # difference average loss between train and val last N epochs
    def diff_avg_loss_lastN(self, N, print_bool = False):
        """
        Calculates the difference between the average validation loss and the average training loss over the last N epochs.

        Parameters:
            N (int): The number of final epochs to consider for the average loss calculation.
            print_bool (bool, optional): If True, prints the difference in average loss; default is False.

        Returns:
            float: The difference between the average validation loss and the average training loss over the last N epochs.

        This method computes the average training and validation loss for the last N epochs, calculates the difference,
        and optionally prints the result based on the evaluation type (either 'pure' or 'augmented').
        """

        diff_avg_loss = round(np.mean(self.val_loss[-N:]) - np.mean(self.loss[-N:]), self.sf)
        
        if print_bool == True:
            print(f"Diff Avg Loss - {self.type}: {diff_avg_loss}")
            
        return diff_avg_loss

    # standard deviation loss from validation data for last N epochs   
    def std_loss_lastN(self, N, print_bool = False):
        """
        Calculates the standard deviation of the validation loss over the last N epochs.

        Parameters:
            N (int): The number of final epochs to consider for the standard deviation calculation.
            print_bool (bool, optional): If True, prints the standard deviation of the validation loss; default is False.

        Returns:
            float: The standard deviation of the validation loss over the last N epochs.

        This method computes the standard deviation of the validation loss for the last N epochs and 
        optionally prints the result based on the evaluation type (either 'pure' or 'augmented').
        """
        
        std_loss = round(np.std(self.val_loss[-N:]), self.sf)
        
        if print_bool == True:
            print(f"Std Loss - {self.type}: {std_loss}")
            
        return std_loss
    
    # confusion matrix and classification report
    def confusion_matrix_class_report(self, print_bool = False):
        """
        Generates a confusion matrix and classification report for the model's predictions.

        Parameters:
            print_bool (bool, optional): If True, prints the confusion matrix and classification report; default is False.

        Returns:
            tuple: A tuple containing:
                1. A PIL Image of the confusion matrix heatmap.
                2. A dictionary representation of the confusion matrix.
                3. A dictionary containing the classification report.

        This method calculates the confusion matrix and classification report using the true and predicted labels.
        It also generates a heatmap of the confusion matrix, which is converted to a PIL Image. 
        Optionally, it can print the confusion matrix and classification report based on the evaluation type (either 'pure' or 'augmented').
        """
        
        y_pred_argmax = np.argmax(self.y_pred, axis=1)        
        y_true_decoded = self.y_true
        
        # Check if y_true needs conversion from one-hot to class labels
        if len(y_true_decoded[0]) > 1:  # This assumes y_true is a list of arrays
            y_true_decoded = [np.argmax(label) for label in y_true_decoded]
        else:
            y_true_decoded = [label for sublist in y_true_decoded for label in sublist]  # Flatten the list if necessary

        # Generate the confusion matrix
        conf_matrix = confusion_matrix(y_true_decoded, y_pred_argmax)
        conf_matrix_dict = {
            f'{self.class_names[i].split()[0]} (T)': {f'{self.class_names[j].split()[0]} (P)': conf_matrix[i, j] for j in range(len(self.class_names))}
            for i in range(len(self.class_names))
        }        

        # Plotting using seaborn for a nicer-looking confusion matrix
        fig, ax = plt.subplots(figsize=(10, 7))
        # Plot the confusion matrix
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title(f'Confusion Matrix - {self.type}')
        
        # Convert to a PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        conf_matrix_PIL = Image.open(buf)
        
        # generate classifcation report as dictionary
        class_report = classification_report(y_true_decoded, y_pred_argmax, target_names=self.class_names, output_dict=True )

        if print_bool == True:                
            plt.show()                       
            print(f"Classification Report - {self.type}:\n")
            for class_name, metrics in class_report.items():
                metrics_str = ', '.join(f"{metric}: {value:.2f}" for metric, value in metrics.items())
                print(f"{class_name}: {metrics_str}")
             
        plt.close(fig)
   
        return conf_matrix_PIL, conf_matrix_dict, class_report

        
    # AUC per class
    def class_auc(self, print_bool=False):
        """
        Calculates the Area Under the Curve (AUC) for each class.

        Parameters:
            print_bool (bool, optional): If True, prints the AUC for each class; default is False.

        Returns:
            dict: A dictionary with class names as keys and their corresponding AUC values as values.

        This method computes the AUC score for each class using the true labels and predicted probabilities.
        Optionally, it prints the AUC values for each class based on the evaluation type (either 'pure' or 'augmented').
        """

        class_indices = list(range(len(self.class_names)))  # ensures order
        y_true_binarized = label_binarize(self.y_true, classes=class_indices)
        
        aucs = {}
        for i, class_name in enumerate(self.class_names):
            # Calculate AUC for each class and add to dictionary
            class_auc = roc_auc_score(y_true_binarized[:, i], self.y_pred[:, i])
            aucs[class_name] = class_auc
            
        if print_bool == True:
            print(f"AUC per class - {self.type}:\n")
            for class_name, auc in aucs.items():
                print(f"  {class_name}: {round(auc, self.sf)}")
        
        return aucs
    
    
    
    # log metrics to ClearML
    def log_metrics(self, task, logger, N, augmentation_metadata, hyperparameters):
        """
        Logs various evaluation metrics to a logging service and registers them as artifacts for tracking and analysis.

        Parameters:
            task (Task): The ClearML task object for managing experiment logging.
            logger (Logger): The logger object used to report scalar values.
            N (int): Number of final epochs to consider for average loss calculations.
            augmentation_metadata (dict): Metadata about the augmentations applied to the dataset.
            hyperparameters (dict): Hyperparameters used in the model training.

        This method calculates various performance metrics including accuracy, AUC, F1 score, and loss statistics.
        It logs key metrics as scalar values and registers comprehensive metrics and reports as artifacts
        in ClearML for further analysis and tracking.
        """

        # evaluation metrics
        avg_acc, avg_acc_val = self.average_accuracy(False)
        std_acc, std_acc_val = self.std_accuracy(False)
        best_acc, best_acc_val = self.best_accuracy(False)
        last_acc, last_acc_val = self.last_accuracy(False)
        overall_auc = self.overall_auc(False)
        max_f1_score, max_f1_epoch = self.max_f1(False)
        min_loss, min_loss_epoch = self.min_loss(False)
        diff_avg_loss = self.diff_avg_loss_lastN(N,False)
        std_loss = self.std_loss_lastN(N,False)
        conf_matrix_PIL, conf_matrix_dict, class_report = self.confusion_matrix_class_report(False)
        class_auc = self.class_auc(False)
        
        # log key metrics as individual scalars
        logger.report_single_value('Best Train Accuracy', best_acc)
        logger.report_single_value('Best Val Accuracy', best_acc_val)
        logger.report_single_value('Overall AUC (Val)', overall_auc)
        logger.report_single_value('F1 score (Val)', max_f1_score)
        logger.report_single_value('Min Loss (Val)', min_loss)
        logger.report_single_value('Diff Final Loss (Val)', diff_avg_loss)

        # log all metrics as dicionary
        all_metrics = {
            "Metric": [
                "Average Accuracy", "Standard Deviation of Accuracy", "Best Accuracy", "Last Accuracy",
                "Overall AUC", "Maximum F1 Score", "Minimum Loss", 
                "Difference in Average Loss Last N", "Standard Deviation of Loss Last N"
            ],
            "Training": [
                avg_acc, std_acc, best_acc, last_acc, None, None, None, None, None  
            ],
            "Validation": [
                avg_acc_val, std_acc_val, best_acc_val, last_acc_val, overall_auc, 
                max_f1_score, min_loss, diff_avg_loss, std_loss  
            ]
            
            }       

        # register as ClearML artifacts
        task.upload_artifact('METRICS', pd.DataFrame(all_metrics))
        task.upload_artifact('CONFUSION_MATRIX', pd.DataFrame(conf_matrix_dict))
        task.upload_artifact('CLASSIFICATION_REPORT', pd.DataFrame(class_report))
        task.upload_artifact('CLASS_AUC', pd.DataFrame(class_auc.items(), columns=['Class', 'AUC']))
        if self.type == 'augmented':
            task.upload_artifact('AUGMENTATION_METADATA', tools.pretty_print_dict(augmentation_metadata))
        task.upload_artifact('HYPERPARAMETERS', tools.pretty_print_dict(hyperparameters))
        task.upload_artifact('CONFUSION_MATRIX_IMAGE', conf_matrix_PIL)

        print('Sending metrics to clearML...')        
        task.close()

    def get_yolo_metrics_dict(metrics):
        """
        Converts YOLO evaluation metrics into a dictionary with specific key-value pairs.

        Parameters:
            metrics (object): An object containing YOLO evaluation metrics, typically including mAP and F1 scores.

        Returns:
            dict: A dictionary containing rounded evaluation metrics, including:
                - 'mAP_50-95': Mean Average Precision from IoU=0.50 to 0.95.
                - 'mAP_50': Mean Average Precision at IoU=0.50.
                - 'mAP_75': Mean Average Precision at IoU=0.75.
                - 'mAP_per_class': List of mAP from IoU=0.50 to 0.95 for each category.
                - 'f1_per_class': List of F1 scores for each category.
                - 'f1_avg': Average F1 score across all categories.

        This function calculates the average F1 score and uses a utility function to round all metric values
        to three significant figures before storing them in the dictionary.
        """

        f1_avg = np.mean(metrics.box.f1)

        metrics_dict = {
            'mAP_50-95': tools.round_to_3sf(metrics.box.map),     # Mean Average Precision from IoU=0.50 to 0.95
            'mAP_50': tools.round_to_3sf(metrics.box.map50),      # Mean Average Precision at IoU=0.50
            'mAP_75':  tools.round_to_3sf(metrics.box.map75),      # Mean Average Precision at IoU=0.75
            'mAP_per_class':  tools.round_to_3sf(metrics.box.maps), # List of mAP from IoU=0.50 to 0.95 for each category
            'f1_per_class':  tools.round_to_3sf(metrics.box.f1), 
            'f1_avg':  tools.round_to_3sf(f1_avg),   
        }
    
        return metrics_dict