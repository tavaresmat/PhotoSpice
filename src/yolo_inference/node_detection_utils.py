import math
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
    for i, data in components.iterrows(): # saving centers and out-connections of each component
        component_center = bbox_center (data)
        bboxes_centers.append (component_center)
        connections_beginnings:list[np.ndarray] = img_graph.outpoints(data)
        for array in connections_beginnings:
            components_out_points[i].append(array_to_string(array))
    
    return bboxes_centers

def filter_bboxes_overlap_by_confidence(components: pd.DataFrame, max_area: float) -> any: # maybe a collision class would be desirable
    '''returns a list of collisions between bboxc.es in "components" dataframe'''
    to_drop = set()
    n_components = components.shape[0]
    for i in range(n_components):
        for j in range(i+1, n_components):
            c = components
            far_horizontally = (c.at[i,'xmin'] > c.at[j, 'xmax']) or (c.at[j,'xmin'] > c.at[i, 'xmax'])
            far_vertically = (c.at[i, 'ymin'] > c.at[j, 'ymax']) or (c.at[j,'ymin'] > c.at[i, 'ymax'])    
            
            print (f'{i} - {j}: h:{far_horizontally}, v:{far_vertically}')
            if far_vertically or far_horizontally:
                continue
            else:
                d1 = c.at[i,'ymax'] - c.at[j, 'ymin']
                d2 = c.at[j,'ymax'] - c.at[i, 'ymin']
                intersection_height = min(d1, d2) 

                d1 =  c.at[i,'xmax'] - c.at[j, 'xmin']
                d2 = c.at[j,'xmax'] - c.at[i, 'xmin']
                intersection_width = min(d1, d2)

                # intersection is correct just if a box dont contains the other, otherwise is bigger
                intersection = intersection_height* intersection_width
                
                areas = [
                    (c.at[i, 'xmax'] - c.at[i, 'xmin']) * (c.at[i, 'ymax'] - c.at[i, 'ymin']),
                    (c.at[j, 'xmax'] - c.at[j, 'xmin']) * (c.at[j, 'ymax'] - c.at[j, 'ymin']),
                ]

                if (intersection < max_area * min(areas)):
                    continue
    
                if c.at[i, 'confidence'] < c.at[j, 'confidence']:
                    to_drop.add(i)
                elif c.at[i, 'confidence'] > c.at[j, 'confidence']:
                    to_drop.add(j)
    
    c.drop (to_drop, inplace=True)
    c.reset_index(inplace=True)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2'"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    radians = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return math.degrees (radians)

def bfs_anchieved_vertices_and_lesser_node(
                vertex,
                adjacency_list,
                vertex_node
            ):

    visited = np.array([False]*len(adjacency_list), dtype=bool)
    is_visited = lambda vertex: visited[vertex]
    def mark_visited(vertex): visited[vertex] = True
    queue = []
    queue.append (vertex)
    mark_visited(vertex)
    lesser_node = None

    while (len(queue) > 0):
        
        current_point = queue.pop(0)
        current_node = vertex_node[current_point]

        has_node = current_node is not None
        first_valid_node = (lesser_node is None) and has_node

        if first_valid_node:
            lesser_node = current_node
        elif has_node and (lesser_node >current_node ):
            lesser_node = current_node

        neighbors = adjacency_list[current_point]
        for neighbor in neighbors:
            if (not is_visited(neighbor)):
                mark_visited(neighbor)
                queue.append(neighbor)
    
    conex_graph_component = []
    for i,boolean in enumerate(visited):
        if boolean:
            conex_graph_component.append (i)


    return (
        conex_graph_component,
        lesser_node
    )
