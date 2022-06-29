from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import pandas
import torch
import cv2
from srca.image_manipulation.image_graph import ImageGraph
from src.augmentation.bbox_manipulation import plot_inference_bbox
from src.image_manipulation.utils import bbox_center, binarize, point_inside_bbox

POS_ATTR = ['xmin', 'xmax', 'ymin', 'ymax']

class ComponentDetector():
    
    model:any = None
    size:int = None
    __last_predictions:pandas.DataFrame = None
    __last_image:np.ndarray = None

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

    def generate_netlist(self):
        components = self.__last_predictions.copy()
        image = self.__last_image.copy()
        img_graph = ImageGraph(binarized_image=image)

        components['value'] = 0
        for index, data in components.iterrows(): # discovering and saving values
            components.at[index, 'value'] = self.nearest_value (data[POS_ATTR])

        components_out_points = [{} for _,_ in components.iterrows()] 
        bboxes_centers = []

        for index, data in components.iterrows(): # saving centers and out-connections of each component
            component_center = bbox_center (data)
            bboxes_centers.append (component_center)
            connections_beginnings:list[np.ndarray] = img_graph.connections_points(data)
            for array in connections_beginnings:
                components_out_points[index][array_to_string(array)] = None

        max_node = 1 # starts at 1 once 0 is reserved to ground 
        current_node = 0

        # YOU ARE WORKING HERE
        for component_index, component_data in components.iterrows(): # for each component
            for connection_index, connection_sarray in enumerate(
                    components_out_points[component_index].keys()
                ): # and each connection of that component
                current_node = max_node
                for other_component_index, collision_point in \
                    img_graph.list_connected( # ITERATING BY GENERATOR !
                        string_to_array(connection_sarray),
                        component_data
                    ): # so, for each connection connected to the first

                    min_distance = np.Infinity
                    nearest_point = None
                    for sarray in components_out_points[other_component_index].keys():
                        array = string_to_array(sarray)
                        distance = np.linalg.norm(collision_point - array)
                        if distance < min_distance:
                            nearest_point = sarray

                    old_node_value = components_out_points[other_component_index][sarray]
                    if old_node_value is None:
                        components_out_points[other_component_index][sarray] = current_node
                    else:
                        if old_node_value < current_node:
                            current_node = old_node_value
                            # MISSING REPLACEMENT OF ALL OLD VALUES
                        else:
                            components_out_points[other_component_index][sarray] = current_node



                if current_node == max_node:
                    max_node += 1
                
                        
                

        # propagatint nodes ...
        #for index, data in components.iterrows(): ...
        raise NotImplementedError()

    def nearest_value(self, position:pandas.Series):
        return 0
        raise NotImplementedError() 


def array_to_string(array:np.ndarray) -> str:
    strings = []
    for value in array:
        strings.append(str(value))
    return 'x'.join(strings)

def string_to_array(sarray:str) -> np.ndarray:
    list_str = sarray.split('x')
    mapped = map(lambda s: int(s) , list_str)
    return np.array (list(mapped))

