
Drone_moreclass3 - v2 2022-10-25 11:26am
==============================

This dataset was exported via roboflow.com on October 25, 2022 at 2:28 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

It includes 2686 images.
For-task2 are annotated in YOLO v7 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 416x416 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* Random rotation of between -20 and +20 degrees
* Random shear of between -20° to +20° horizontally and -20° to +20° vertically
* Random exposure adjustment of between -25 and +25 percent
* Random Gaussian blur of between 0 and 6.75 pixels


