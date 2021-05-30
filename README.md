# ProtoPShare

This code package implements the prototypical sharing part network (ProtoPShare)
from the paper "ProtoPShare: Prototypical Parts Sharing for Similarity Discovery in Interpretable Image Classification"
(to appear at KDD 2021), by Dawid Rymarczyk (Jagiellonian University), Łukasz Struski (Jagiellonian University),
Jacek Tabor (Jagiellonian University), and Bartosz Zieliński (Jagiellonian University).

This code package was based upon ProtoPNet from https://github.com/cfchen-duke/ProtoPNet. 

Prerequisites: PyTorch, NumPy, cv2, Augmentor (https://github.com/mdbloice/Augmentor)
Recommended hardware: 1 NVIDIA Tesla V-100 GPUs

Instructions for preparing the data:
1. Download the dataset CUB_200_2011.tgz from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
2. Unpack CUB_200_2011.tgz
3. Crop the images using information from bounding_boxes.txt (included in the dataset)
4. Split the cropped images into training and test sets, using train_test_split.txt (included in the dataset)
5. Put the cropped training images in the directory "./datasets/cub200_cropped/train_cropped/"
6. Put the cropped test images in the directory "./datasets/cub200_cropped/test_cropped/"
7. Augment the training set using img_aug.py (included in this code package)
   -- this will create an augmented training set in the following directory:
      "./datasets/cub200_cropped/train_cropped_augmented/"
8. Train a ProtoPNet model according to https://github.com/cfchen-duke/ProtoPNet
9. Update settings file to start the pruning. 
9. Run main.py
