from ctypes.wintypes import PDWORD
import pandas as pd

import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as alb

def binarize (image:np.array, kernel_dimension:tuple=None) -> np.array:
    """
        Converts the RBG image given to a binary image that aims to 
        mark contours and drawings as 1 and background as 0. Returns
        such contours image.

        args:
            image (np.array): image whose contours will be detected 
            kernel_dimension (tuple): kernel for morphological closing operation 
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5,5), 2)
    threshold_image = cv2.Canny (blurred_image, 10, 30) 
    
    if kernel_dimension is None:
        kernel_dimension = (
            2, #+ (image.shape[0] // 300),
            2 #+ (image.shape[0] // 200)
        )

    kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_dimension)
    closed_threshold = cv2.morphologyEx (threshold_image, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed_threshold

def open_rgb_image (path):
    cv2_img = cv2.imread(path)
    return cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

def point_inside_bbox (point:np.ndarray, bbox:pd.Series):
        return (point[0] > bbox['xmin']) and (point[0] < bbox['xmax']) \
        and (point[1] > bbox['ymin']) and (point[1] < bbox['ymax'])

def bbox_center (bbox: pd.Series):
    return .5 * np.array([
        bbox['xmin'] + bbox['xmax'],
        bbox['ymin'] + bbox['ymax']
    ])