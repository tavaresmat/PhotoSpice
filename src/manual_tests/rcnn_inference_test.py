from sys import float_repr_style
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.image_manipulation.utils import binarize
from src.yolo_inference.keypoints_net import KeypointsNet

photo_path = f'd2.png'
keypoints_detector = KeypointsNet('models/diode keypoints.pth')

def padding():
    
    image = cv2.cvtColor( cv2.imread(photo_path), cv2.COLOR_BGR2GRAY )
    long_image = np.zeros([image.shape[0]*2, image.shape[1]*2], image.dtype)
    long_image[0:image.shape[0], 0:image.shape[1]] = image
    image = long_image

    print (keypoints_detector(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)))
    keypoints_detector.predict_and_plot(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))

def normal_test():
    image = cv2.cvtColor( cv2.imread(photo_path), cv2.COLOR_BGR2GRAY )

    print (keypoints_detector(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)))
    keypoints_detector.predict_and_plot(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))

padding()