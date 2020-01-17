# MouseFree: Computer Vision Based Mouse

## Goal: 

Hand image and motion recognition for controlling cursor placement and actions on computer. Meant to create a mouse-free computer environment. 

Actions supported: moving cursor, single left click, double left click, drag left click

## General layout:

- Deep convolutional neural network (CNN) trained on hand images. Used output layer of 4 labels with softmax and cross entropy loss
- Detection with OpenCV by extracting contours, background subtraction, convex hull extraction, erosion and blurring for CNN training and detection
- Use of PyAutoGui to relay detected information to mouse cursor control

## Results:

- Training of deep model resulted in relative inability to detect Palms. Required special treatment and planning to find optimal model.
- 98% accuracy in the test set 
- Smooth transitions between motion by adding velocity and momentum term to detection scheme


