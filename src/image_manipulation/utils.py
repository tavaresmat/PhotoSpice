from ctypes.wintypes import PDWORD
import matplotlib
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
    CLOSINGS = 1

    plt.style.use('ggplot')
    matplotlib.use( 'tkagg' )

    for i in range(15):
        cut = analyzed[i*shape[0]//15 - (1 if i != 0 else -1) , 0:shape[1]]
        found_edge, found_valley = False, False
        width = 0
        
        for j in range (shape[1]):
            if (cut[j] != 0):
                found_edge = True
            if (found_edge and cut[j] == 0):
                width += 1
                found_valley = True
            if (found_valley and cut[j] != 0):
                if (width <= shape[1]//20):
                    calculated_widths.append(width)
                    width = 0
                    found_valley = False

    #print (f'widths: {calculated_widths}')
    kernel_size = np.median(calculated_widths) if calculated_widths else kernel_size

    for i in range(DILATIONS):
        size = max (round(kernel_size*0.5), 2)
        image = cv2.dilate (
            image,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size,)*2)
        )

    for i in range(CLOSINGS):
        image = cv2.morphologyEx(
            image, 
            cv2.MORPH_CLOSE, 
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (round(kernel_size),)*2),
        )

    return image, kernel_size

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

def downgrade_image (img: np.ndarray, max_pixels:int) -> np.ndarray:
        y, x = img.shape[:2]
        scale = (max_pixels/(x*y))**0.5
        if max_pixels < (x*y):
            img = cv2.resize(img, np.array([x*scale, y*scale]).astype(int) )
        return img