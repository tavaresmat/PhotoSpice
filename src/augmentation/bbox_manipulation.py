from random import randint
from cv2 import norm

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

from ..image_manipulation.utils import open_rgb_image

__class_color = {}
randcolor = lambda: (randint(0,255), randint(0,255), randint(0,255))

def load_bbox_list(txtpath:str) -> None:
    """
    Loads a bbox txt file in yolo format and converts
    into a list of lists, seeking "albumentations" module
    model

    Args:
        txtpath (str): path of txt bboxfile that will be loaded
    """
    boxes = []
    with open(txtpath, 'r') as txtfile:
        for line in txtfile:
            line_words = line.split(' ')
            label = line_words.pop(0)
            values = list(map (
                float,
                line_words
            ))
            values.append (label)
            boxes.append(values)
    return boxes

def save_bbox_list (filepath:str, bbox_list:"list[list[any]]")-> None:
    """
    Save a bbox txt file in yolo format, reading from
    a list of lists according to "albumentations" module
    model. Does the exact opposite of "load_bbox_list".

    Args:
        filepath (str): path of txt bboxfile to be created and saved
        bbox_list (list[list[any]]): list of lists representing bbox
            classification positions and labels
    """
    filetext = ""
    for _tup in bbox_list:
        line_values = list(_tup)
        class_label = line_values.pop(-1)
        filetext += (f"{class_label} ")
        for value in line_values:
            filetext += (f"{value} ")
        filetext = filetext[:-1] + '\n'
    filetext = filetext[:-1] 
    with open (filepath, 'w') as file:
        file.write(filetext)


def draw_yolo_bbox (img:np.array, bbox_list:"list[list[any]]")-> None:
    """
    Draw the bbox list information on the image

    Args:
        img (np.array): image that will receive "bbox_list" information
            graphically over it
        bbox_list (list[list[any]]): list of lists representing bbox
            classification positions and labels
    """
    dh, dw, _ = img.shape

    for values in bbox_list:
        try:
            x, y, w, h, class_label = values
        except ValueError:
            x, y, w, h = values # if have not a class
            class_label = 'None'

        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)
        
        if l < 0:
            l = 0
        if r > dw - 1:
            r = dw - 1
        if t < 0:
            t = 0
        if b > dh - 1:
            b = dh - 1

        color = load_or_create_color (class_label)
        cv2.rectangle(img, (l, t), (r, b), color, dw//300)
        font_scale, font_thickness = 0.3, 2
        ((text_width, text_height), _) = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.putText(img, class_label, (int(l), int(t - 0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                font_thickness)

def plot_bbox_from_path (imgpath:str, labelspath:str)->None:
    """
    Plot the bbox list information over the image given

    Args:
        imgpath (str): path of a background to "bbox_list" information
            be displayed graphically over it
        labelspath (str): bbox txt file in yolo format containing
            classification positions and labels to be plotted
    """
    img = open_rgb_image(imgpath)
    draw_yolo_bbox (img, load_bbox_list(labelspath) )
    plt.imshow (img)
    plt.show()

def plot_bbox_from_object(img:np.array, bboxes:"list[list[any]]")->None:
    """
    Plot the bbox list information over the image given

    Args: 
        img (np.array): image that will be a background to "bbox_list"
            information to be displayed graphically over it
        bboxes (list[list[any]]): list of lists representing bbox
            classification positions and labels to be plotted
    """
    image = img.copy()
    draw_yolo_bbox(image, bboxes )
    plt.imshow (image)
    plt.show()

def load_or_create_color(class_label):
    if class_label not in __class_color.keys():
            similar_color = True
            random_color = randcolor()
            '''while(similar_color):
                similar_color = False
                random_color = randcolor()
                for old_color in __class_color.values():
                    if norm (
                        np.array(random_color) -
                        np.array(old_color)
                    ) < 150:
                        similar_color = True
                        break'''

            __class_color[class_label] = random_color
                        
    return __class_color[class_label]

def draw_inference_bbox (img:np.array, pred_dataframe:"pd.DataFrame", fontsize=None, thickness=None)-> None:
    """
    Draw the bbox list information on the image

    Args:
        img (np.array): image that will receive "bbox_list" information
            graphically over it
        bbox_list (list[list[any]]): list of lists representing bbox
            classification positions and labels
    """

    if fontsize is None:
        fontsize = img.shape[1]/1150
    if thickness is None:
        thickness = img.shape[1]//300

    for _, prediction in pred_dataframe.iterrows():
            color = load_or_create_color (prediction['name'])
            values = {}
            for field in ['xmin', 'ymin', 'xmax', 'ymax']:
                values[field] = int (prediction[field])

            cv2.rectangle(img, (values['xmin'], values['ymin']), (values['xmax'], values['ymax']), color, thickness)
            font_scale, font_thickness = fontsize, thickness
            ((text_width, text_height), _) = cv2.getTextSize(prediction['name'], cv2.FONT_HERSHEY_TRIPLEX, font_scale, thickness)
            ytext = int(values['ymin'] - 0.3 * text_height)
            if ytext - text_height < 0 :
                ytext = int(values['ymax'] + 1.2 * text_height)

            cv2.putText(
                img,
                f"{prediction['name']}({ str(prediction['confidence']*100)[:5] }%)",
                (
                    values['xmin'],
                    ytext
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (*color, 125),
            font_thickness)

def plot_inference_bbox (img:np.array, pred_dataframe:"pd.DataFrame", fontsize=None, thickness=None)->None:
    """
    Plot the bbox list information over the image given

    Args: 
        img (np.array): image that will be a background to "bbox_list"
            information to be displayed graphically over it
        bboxes (list[list[any]]): list of lists representing bbox
            classification positions and labels to be plotted
    """
    try:
        plt.style.use('ggplot')
        matplotlib.use( 'tkagg' )
    finally:
        image = img.copy()
        draw_inference_bbox(image, pred_dataframe, fontsize, thickness)
        plt.imshow (image)
        plt.show()