# Intruder-Aircraft-Detection

Project Briefing - The Importance of Data Augmentation for
Intruder Aircraft Detection
Tom Evans
Supervisor: Son Hoang
November 2, 2023

1 Introduction
This project will investigate the importance of data augmentation when detecting intruder aircraft
from the point of view of another. I intend to use a publicly available data set containing various
images of intruder aircraft to train a machine-learning object detection algorithm to detect and
classify these aircraft. I will then use data augmentation image processing algorithms to create a
larger data set to train the algorithm to see if this improves its performance and effectiveness.

2 Problems
• An object detection algorithm must be trained so that it can effectively locate and classify
an intruder aircraft within an image
• An augmented data set must be created using data augmentation and image transformations
from the original data set
• The original data set must be publicly available to use
• Must be able to compare the results of the object detection algorithm when trained on the
separate data sets

3 Goals
• To train an object detection algorithm to effectively detect intruder aircraft within still images
• To use data augmentation algorithms to create a second larger data set to train the object
detection algorithm with
• To compare and interpret the results of the object detection algorithm with the two data
sets and report them.

4 Scope
• The project will use a publicly available data set [1]
• The project will use a publicly available object detection algorithm, e.g., [2]
• The project will use existing data augmentation Python libraries, e.g., [3]


References
[1] Smyers, E., Katz, S., Corso, A., and Kochenderfer, M. (2023). AVOIDDS: A dataset for vision-
based aircraft detection. Stanford Digital Repository.
Available at https://purl.stanford.edu/hj293cv5980. https://doi.org/10.25740/hj293cv5980.
[2] https://github.com/meituan/YOLOv6
[3] https://pypi.org/project/metamorphic-relations
