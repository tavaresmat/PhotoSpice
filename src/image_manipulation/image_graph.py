
import numpy as np

class ImageGraph:
    '''
    Describes a graph where each vertex is a image pixel, and adjacent pixels
    which have the same color are linked by an edge.
    '''

    def __init__(self, image:np.ndarray): 
        raise NotImplementedError()