from argparse import ArgumentError
from asyncio.proactor_events import _ProactorBaseWritePipeTransport
from queue import Queue
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import cv2

from src.image_manipulation.utils import binarize

class ImageGraph:
    '''
    Describes a graph where each vertex is a image pixel, and adjacent pixels
    which have the same color are linked by an edge.
    '''
    binarized_image = None
    def __init__(self, image:np.ndarray=None, binarized_image:np.ndarray=None): 
        if (image is None) and (binarized_image is None):
            raise ArgumentError(
                'you must provide either "image" or "binarized_image" argument'
            )
        if binarized_image is None:
            binarized_image = binarize(image) 
        self.binarized_image = binarized_image
            
    def outpoints(self, bbox_data:pd.Series) -> list[np.ndarray]:
        '''
        Lists all not-connected points of image that intersects with 
        the bounding box described in "bbox-data"
        '''
        xmin, xmax, ymin, ymax = bbox_data[['xmin', 'xmax', 'ymin', 'ymax']].astype(int)
        borders = {}
        for key in ['top_border', 'bottom_border', 'left_border', 'right_border']:
            borders[key] = self.binarized_image * 0

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
                outpoints_centers.append(center)
        
        return outpoints_centers

    def bfs_bbox_collisions(self, start_point:np.ndarray, bboxes_dataframe:pd.DataFrame) -> tuple[int, np.ndarray]:
        '''
        return index of components connected in image to "start_point", according to 
        "bboxes_dataframe" data, and also returns the point of collision between the search and the bbox
        '''
        visited = self.binarized_image * 0

        is_visited = lambda ndarray: visited[ndarray[0], ndarray[1]]
        def mark_visited(ndarray): visited[ndarray[0], ndarray[1]] = 1

        queue = Queue()
        queue.put (start_point)
        mark_visited(start_point)
        while (queue.not_empty):
            searched_vertex = queue.get()

            # check if seached_vertex is inside some bbox,
            # if so, does not allow its neighbors to be visited
            # and add it to a list, as its a interest point

            for neighbor in self.neighbors_of(searched_vertex):
                if (not is_visited(neighbor)):
                    mark_visited(neighbor)
                    queue.put(neighbor)

        raise NotImplementedError()

    def neighbors_of(self, point:np.ndarray) -> list[np.ndarray]:
        neighbors = []
        candidates = [
            point + np.array([1,0]),
            point - np.array([1,0]),
            point + np.array([0,1]),
            point - np.array([0,1]),
        ]
        for candidate in candidates:
            try:
                color = self.binarized_image[candidate[0], candidate[1]]
            except IndexError: pass
            if color != 0:
                neighbors.append (candidate)
        return neighbors