import numpy as np
import pandas as pd

from src.image_manipulation.utils import bbox_center

def array_to_string(array:np.ndarray) -> str:
    strings = []
    for value in array:
        strings.append(str(value))
    return 'x'.join(strings)

def string_to_array(sarray:str) -> np.ndarray:
    list_str = sarray.split('x')
    mapped = map(lambda s: int(s) , list_str)
    return np.array (list(mapped))

def bboxes_centers_and_fill_outpoints (components, img_graph, components_out_points):
    '''fill "components_out_points" with outpoints positions as string-keys and return bboxes centers'''
    bboxes_centers = []
    for index, data in components.iterrows(): # saving centers and out-connections of each component
        component_center = bbox_center (data)
        bboxes_centers.append (component_center)
        connections_beginnings:list[np.ndarray] = img_graph.outpoints(data)
        for array in connections_beginnings:
            components_out_points[index].append(array_to_string(array))
    
    return bboxes_centers

def bboxes_collisions(components: pd.DataFrame) -> list[any]: # maybe a collision class would be desirable
    '''returns a list of collisions between bboxes in "components" dataframe'''
    raise NotImplementedError()

def filter_collision(components: pd.DataFrame, collision: any) -> pd.DataFrame:
    '''returns "components" dataframe filtering by confidence collided components'''
    raise NotImplementedError()

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def bfs_anchieved_vertices_and_lesser_node(
                vertex,
                adjacency_list,
                vertex_node
            ):
    raise NotImplementedError()