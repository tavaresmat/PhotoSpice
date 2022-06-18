from matplotlib import pyplot as plt
import matplotlib
import torch
import cv2
from src.augmentation.bbox_manipulation import plot_inference_bbox
from src.image_manipulation.utils import binarize

class ComponentDetector():
    
    model = None
    size = None
    __last_predictions = None
    __last_image = None

    def __init__(self, size=600, weights='models/best 400ep map.91.pt'):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, device='cpu')
        self.size = size

    def predict(self, img):
        if type(img) is str:
            img = cv2.imread(img)

        results = self.model(binarize(img), self.size)  
        self.__last_image = img.copy()
        self.__last_predictions = results.pandas().xyxy[0]
        return self.__last_predictions

    def predict_binarized(self, img):
        self.__last_image = img.copy()
        self.__last_predictions = self.model(binarize(img), self.size).pandas().xyxy[0]
        return self.__last_predictions

    def plot(self, img=None, boxes=None):
        if img is None:
            img = self.__last_image
        if boxes is None:
            boxes = self.__last_predictions

        plot_inference_bbox(img, boxes)

    def show_binarized(self):
        try:
            plt.style.use('ggplot')
            matplotlib.use( 'tkagg' )
        finally:
            plt.imshow(binarize(self.__last_image))
            plt.show()