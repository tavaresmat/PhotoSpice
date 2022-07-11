from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import pandas
import torch
import cv2

from src.augmentation.bbox_manipulation import plot_inference_bbox
from src.image_manipulation.utils import binarize
class ComponentDetector():
    
    model:any = None
    size:int = None
    last_predictions:pandas.DataFrame = None
    last_image:np.ndarray = None
    last_binarized: np.ndarray = None

    def __init__(self, size=600, weights='models/components mAP.97 close2x1 400ep.pt'):
        
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, device='cpu')
        self.size = size

    def predict(self, img) -> pandas.DataFrame:
        if type(img) is str:
            img = cv2.imread(img)

        
        '''y, x = img.shape[:2]
        scale = (MAX_PIXELS/(x*y))**0.5
        if MAX_PIXELS < (x*y):
            img = cv2.resize(img, np.array([x*scale, y*scale]).astype(int) )'''

        self.last_image = img.copy()
        self.last_binarized = binarize(img.copy())
        results = self.model(self.last_binarized, self.size)  
        self.last_predictions = results.pandas().xyxy[0]

        return self.last_predictions

    def predict_binarized(self, img):
        self.last_binarized = img.copy()
        self.last_image = None
        self.last_predictions = self.model(binarize(img), self.size).pandas().xyxy[0]
        return self.last_predictions

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