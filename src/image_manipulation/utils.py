import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as alb

def binarize (image:np.array) -> np.array:
    """
        Converts the RBG image given to a binary image that aims to 
        mark contours and drawings as 1 and background as 0. Returns
        such contours image.

        args:
            image (np.array): image whose contours will be detected 
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5,5), 1)
    threshold_image = cv2.Canny (blurred_image, 15, 40) 
    kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    closed_threshold = cv2.morphologyEx (threshold_image, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed_threshold

def open_rgb_image (path):
    cv2_img = cv2.imread(path)
    return cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)