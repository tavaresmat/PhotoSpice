from argparse import ArgumentError
from asyncio.proactor_events import _ProactorBaseWritePipeTransport
from math import ceil
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import cv2

from src.image_manipulation.utils import binarize, point_inside_bbox

class ImageGraph:
    '''
    Describes a graph where each vertex is a image pixel, and adjacent pixels
    which have the same color are linked by an edge.
    '''
    binarized_image = None
    global_visited_vertices = None
    def __init__(self, image:np.ndarray=None, binarized_image:np.ndarray=None): 

        try:
            plt.style.use('ggplot')
            matplotlib.use( 'tkagg' )
        finally:
            if (image is None) and (binarized_image is None):
                raise ArgumentError(
                    'you must provide either "image" or "binarized_image" argument'
            )
        

        if binarized_image is None:
            binarized_image = binarize(image) 
        self.binarized_image = binarized_image
        self.global_visited_vertices = binarized_image * 0 
            
            
    def outpoints(self, bbox_data:pd.Series) -> list[np.ndarray]:
        '''
        Lists all not-connected points of image that intersects with 
        the bounding box described in "bbox-data"
        '''
        xmin, xmax, ymin, ymax = bbox_data[['xmin', 'xmax', 'ymin', 'ymax']].astype(int)
        borders = {}
        for key in ['top_border', 'bottom_border', 'left_border', 'right_border']:
            borders[key] = self.binarized_image * 0

        ymax -= 1 if ymax >= self.binarized_image.shape[0] else 0
        xmax -= 1 if xmax >= self.binarized_image.shape[1] else 0
        ymin += 1 if ymin < 0 else 0
        xmin += 1 if ymin < 0 else 0

        borders['top_border'][ymin,xmin:xmax] = self.binarized_image[ymin,xmin:xmax]
        borders['bottom_border'][ymax,xmin:xmax] = self.binarized_image[ymax,xmin:xmax]
        borders['left_border'][ymin:ymax, xmin] = self.binarized_image[ymin:ymax, xmin]
        borders['right_border'][ymin:ymax, xmax] = self.binarized_image[ymin:ymax, xmax]

        outpoints_centers = []

        for border in borders.values():
            contours, _ = cv2.findContours(border,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                points = 0
                sum = np.array([0,0])
                for point in c:
                    sum += np.array(point[0])
                    points += 1
                center = ( sum * (1/points) ).astype(int)
                outpoints_centers.append(center[::-1])
        
        return outpoints_centers

    def bfs_bbox_collisions(self, start_point:np.ndarray, bboxes_dataframe:pd.DataFrame) -> tuple[int, np.ndarray]:
        '''
        return index of components connected in image to "start_point", according to 
        "bboxes_dataframe" data, and also returns the point of collision between the search and the bbox
        '''
        visited = self.binarized_image * 0
        is_visited = lambda ndarray: visited[ndarray[0], ndarray[1]]
        def mark_visited(ndarray): visited[ndarray[0], ndarray[1]] = 1

        queue = []
        try:
            queue.append (start_point)
            mark_visited(start_point)
        except IndexError:
            print ('WARNING: start point out of image')
            return

        once_out_bbox = False
        while (len(queue) > 0):
            current_point = queue.pop(0)

            inside_bbox = False
            for comp_index, component_bbox in bboxes_dataframe.iterrows():
                if (point_inside_bbox(current_point, component_bbox)): 
                    inside_bbox = True
                    mark_visited(current_point)
                    yield (comp_index, current_point)
                    break

            if (not once_out_bbox) and (not inside_bbox):
                once_out_bbox = True
                visited = self.global_visited_vertices
                mark_visited(current_point)
                queue = [current_point]

            if not (inside_bbox and once_out_bbox): 
                neighbors = self.neighbors_of(current_point)
                for neighbor in neighbors:
                    if (not is_visited(neighbor)):
                        mark_visited(neighbor)
                        queue.append(neighbor)
        '''plt.imshow (visited * 100 + self.binarized_image)
        plt.show()'''

    def neighbors_of(self, point:np.ndarray) -> list[np.ndarray]:
        neighbors = []
        y, x = self.binarized_image.shape
        stepx, stepy = ceil(x/200), ceil(y/200) 
        candidates = [
            point + np.array([stepy,0]),
            point - np.array([stepy,0]),
            point + np.array([0,stepx]),
            point - np.array([0,stepx]),
        ]
        for candidate in candidates:
            valid_index = (candidate[0] >= 0) \
                and (candidate[1] >= 0) \
                and (candidate[0] < self.binarized_image.shape[0]) \
                and (candidate[1] < self.binarized_image.shape[1])
            if (valid_index):
                color = self.binarized_image[candidate[0], candidate[1]]
                if color != 0:
                    neighbors.append (candidate)
        return neighbors