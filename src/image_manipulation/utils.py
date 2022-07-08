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

def inflate(image, analyzed=None):
    assert (len(image.shape) == 2)

    if analyzed is None:
        analyzed = image.copy()

    shape = image.shape
    kernel_size = 5
    calculated_widths = []
    DILATIONS = 1
    CLOSINGS = 2

    for i in range(15):
        cut = analyzed[i*shape[0]//15 - (1 if i != 0 else 0) , 0:shape[1]]
        found1 = False, False
        width = 0
        for j in range (shape[1]):
            if (cut[j] != 0):
                found1 = True
            if (found1 and cut[j] == 0):
                width += 1
            if (found1 and cut[j] != 0 and width > 0):
                if (width <= shape[1]//20):
                    calculated_widths.append(width)
                    break

    kernel_size = np.median(calculated_widths) if calculated_widths else kernel_size

    for i in range(DILATIONS):
        image = cv2.dilate (
            image,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (round(kernel_size*.7),)*2)
        )

    for i in range(CLOSINGS):
        image = cv2.morphologyEx(
            image, 
            cv2.MORPH_CLOSE, 
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (round(kernel_size*1.3),)*2),
        )

    return image

def open_rgb_image (path):
    cv2_img = cv2.imread(path)
    return cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

def point_inside_bbox (point:np.ndarray, bbox:pd.Series):
        return (point[1] > int(bbox['xmin'])) and (point[1] < int(bbox['xmax'])) \
        and (point[0] > int(bbox['ymin'])) and (point[0] < int(bbox['ymax']))

def bbox_center (bbox: pd.Series):
    return .5 * np.array([
        bbox['ymin'] + bbox['ymax'],
        bbox['xmin'] + bbox['xmax']
    ])