
from argparse import ArgumentError
import numpy as np

from image_manipulation.utils import binarize

class ImageGraph:
    '''
    Describes a graph where each vertex is a image pixel, and adjacent pixels
    which have the same color are linked by an edge.
    '''

    def __init__(self, image:np.ndarray=None, binarized_image:np.ndarray=None): 

        if (image is None) and (binarized_image is None):
            raise ArgumentError(
                'you must provide either "image" or "binarized_image" argument'
            )
        
        if binarized_image is None:
            binarized_image = binarize(image) 
            
        raise NotImplementedError()