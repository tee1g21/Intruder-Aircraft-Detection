# Intruder-Aircraft-Detection

The Importance of Data Augmentation for Image Classification and Intruder Aircraft DetectionIntruder Aircraft Detection
Tom Evans
June 7, 2024

1.1 Problem
When identifying intruder aircraft from the perspective of an opposing aircraft,
both image classification and object detection systems need to be robust and
consistently perform at their maximum potential. Data augmentation presents
a promising approach for enhancing performance, particularly when training
data is limited. However, research that directly compares the effects of data
augmentation on image classification versus object detection is scarce. It is
not yet clear whether augmentation techniques that significantly improve image
classification also boost object detection to a comparable degree, highlighting
the need for investigation.


1.2 Goals
The goal of this project is to examine whether the most effective data augmentation techniques from image classification can enhance object detection performance, particularly for detecting intruder aircraft. This will be explored using
samples of varying sizes from the AVOIDDS dataset [1], which simulates scenarios with limited training data where data augmentation would be more beneficial.
Furthermore, the project will compare Image Classification performance using
equivalently sized non-augmented and augmented training sets to determine if
augmentation leads to better results when the number of training data points is
consistent.


1.3 Scope
This project will investigate Data Augmentation with Image Classification, with
different methods and percentages of Augmentation. Only the most effective
augmentation configurations will be applied to object detection. The research
will employ a custom model for image classification, constructed using TensorFlow’s Keras framework, and will leverage a pre-trained YOLO model for object
detection.

