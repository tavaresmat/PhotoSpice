
from argparse import ArgumentError
from email.generator import Generator
import numpy as np
import pandas as pd

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

        self.binarized_image ()
        

        raise NotImplementedError()

    def bfs_bbox_collisions(self, start_point:np.ndarray, bboxes_dataframe:pd.DataFrame) -> tuple[int, np.ndarray]:
        '''
        return index of components connected in image to "start_point", according to 
        "bboxes_dataframe" data, and also returns the point of collision between the search and the bbox
        '''

        raise NotImplementedError()