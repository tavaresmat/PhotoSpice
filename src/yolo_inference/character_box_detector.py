from math import inf
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import pandas
import torch
import cv2

from src.augmentation.bbox_manipulation import normalize_coords, plot_inference_bbox
from src.image_manipulation.utils import bbox_center, binarize
from src.yolo_inference.node_detection_utils import filter_bboxes_overlap_by_confidence

MAX_OVERLAP_AREA = 0.60

class CharacterBoxDetector():
    
    model:any = None
    size:int = None
    last_predictions:pandas.DataFrame = None
    last_image:np.ndarray = None
    last_binarized: np.ndarray = None

    def __init__(self, size=800, weights='models/characters_mAP.87.pt'):
        
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, device='cpu')
        self.size = size
    
    def predict(self, img) -> pandas.DataFrame:
        if type(img) is str:
            img = cv2.imread(img)

        self.last_image = img.copy()
        self.last_binarized = binarize(img)
        return self.predict_binarized(
            self.last_binarized
        )

    def predict_binarized(self, img) -> pandas.DataFrame:
        self.last_binarized = img
        self.last_predictions = self.model(
            self.last_binarized, 
            self.size
        ).pandas().xyxy[0]
        return self.last_predictions

    def group_characters(self, img) -> pandas.DataFrame:
        chars = self.predict_binarized(img) # AT THIS TIME ITS BINARIZED
        filter_bboxes_overlap_by_confidence(
            chars,
            MAX_OVERLAP_AREA
        )
        chars = chars.sort_values('xmin')
        chars.reset_index(inplace=True)

        chars['xcenter'] = chars['ycenter'] = None
        for index, char in chars.iterrows():
            center = bbox_center (char)
            chars.at[index, "ycenter"] = center[0]
            chars.at[index, "xcenter"] = center[1]

        indexes = list(chars.index)
        current_string = ''
        charboxes = pandas.DataFrame(columns=['string', 'xcenter', 'ycenter'])

        while ( len(indexes) > 0 ):
            xcenter = ycenter = char_number = 0
            current_char = chars.iloc[indexes[0]]
            current_string += current_char['name']
            xcenter += current_char['xcenter']
            ycenter += current_char['ycenter']
            xmin = current_char['xmin']
            xmax = current_char['xmax']
            char_number += 1
            del indexes[0]

            next_char, next_index = next_right_box (current_char, chars) 
            while (next_char is not None):
                indexes.remove (next_index)
                xcenter += next_char['xcenter'] # will turn into an average at the final
                ycenter += next_char['ycenter'] # will turn into an average at the final
                char_number += 1
                current_string += next_char['name']
                xmax = next_char['xmax']
                next_char, next_index = next_right_box (next_char, chars) 
            new_string_df = pandas.DataFrame({
                    'string': [current_string],
                    'xcenter': [int(xcenter/char_number)], # that's an average
                    'ycenter':  [int(ycenter/char_number)], # that's an average
                    'xmin': [int(xmin)],
                    'xmax': [int(xmax)],
                    'ymin': 0.0, #not implemented
                    'ymax': 0.0 #not implemented
            })
            charboxes = pandas.concat([charboxes, new_string_df])
            current_string = ''
        
        return normalize_coords(charboxes.reset_index(), img.shape)
            

    def plot(self, img=None, boxes=None):
        if img is None:
            img = self.last_image if self.last_image is not None else self.last_binarized
        if boxes is None:
            boxes = self.last_predictions

        plot_inference_bbox(img, boxes)

    def show_binarized(self):
        try:
            plt.style.use('ggplot')
            matplotlib.use( 'tkagg' )
        finally:
            plt.imshow(self.last_binarized)
            plt.show()

def next_right_box(this_box, boxes, link_distance=None):
    xmin, xmax, ymin, ymax = this_box[
        ['xmin', 'xmax', 'ymin', 'ymax']
    ]
    width = xmax - xmin
    candidates = boxes[
        (boxes['xcenter'] > this_box['xcenter']) &
        (boxes['xcenter'] < this_box['xcenter'] + width*2.5) &
        (boxes['ycenter'] < ymax) & 
        (boxes['ycenter'] > ymin)
    ]
    if candidates.shape[0] == 0:
        return None, None
    elif candidates.shape[0] == 1:
        return candidates.iloc[0], candidates.index[0]
    else:
        #raise NotImplementedError(f'Precisamos de um criterio de desempate neste caso:\n{candidates}')
        minor_xcenter = inf
        choosen_char = None
        choosen_char_index = None
        for index, char in candidates.iterrows():
            if char['xcenter'] < minor_xcenter:
                minor_xcenter = char['xcenter']
                choosen_char = char
                choosen_char_index = index

        return choosen_char, choosen_char_index

