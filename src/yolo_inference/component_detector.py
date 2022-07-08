from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import pandas
import torch
import cv2
from math import pi, inf

from src.image_manipulation.image_graph import ImageGraph
from src.augmentation.bbox_manipulation import plot_inference_bbox
from src.image_manipulation.utils import bbox_center, binarize, inflate, point_inside_bbox
from src.yolo_inference.node_detection_utils import (
        array_to_string,
        bboxes_collisions,
        string_to_array,
        bboxes_centers_and_fill_outpoints,
        filter_collision,
        angle_between,
        bfs_anchieved_vertices_and_lesser_node
)

MINIMUM_LINKING_NODE_ANGLE = 100 *(pi/180)
POLARIZED_COMPONENTS = ['diode', 
'voltage', 
#'signal'
]
POS_ATTR = ['xmin', 'xmax', 'ymin', 'ymax']

COMPONENT_LETTER = {
    'diode': 'D',
    'resistor': 'R',
    'inductor': 'L',
    'capacitor': 'C',
    'voltage': 'V',
    'signal': 'Vsin'
}

class ComponentDetector():
    
    model:any = None
    size:int = None
    __last_predictions:pandas.DataFrame = None
    __last_image:np.ndarray = None
    __last_binarized: np.ndarray = None

    def __init__(self, size=600, weights='models/best 400ep map.91.pt'):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, device='cpu')
        self.size = size

    def predict(self, img):
        if type(img) is str:
            img = cv2.imread(img)

        self.__last_image = img.copy()
        self.__last_binarized = binarize(img.copy())
        results = self.model(self.__last_binarized, self.size)  
        self.__last_predictions = results.pandas().xyxy[0]
        return self.__last_predictions

    def predict_binarized(self, img):
        self.__last_binarized = img.copy()
        self.__last_image = None
        self.__last_predictions = self.model(binarize(img), self.size).pandas().xyxy[0]
        return self.__last_predictions

    def plot(self, img=None, boxes=None):
        if img is None:
            img = self.__last_image if self.__last_image is not None else self.__last_binarized
        if boxes is None:
            boxes = self.__last_predictions

        plot_inference_bbox(img, boxes)

    def show_binarized(self):
        try:
            plt.style.use('ggplot')
            matplotlib.use( 'tkagg' )
        finally:
            plt.imshow(self.__last_binarized)
            plt.show()

    def generate_netlist(self):
        components = self.__last_predictions.copy() # components list
        image = self.__last_binarized.copy()
        image = inflate(image)
        img_graph = ImageGraph(binarized_image=image)

        components_outpoints = [[] for _,_ in components.iterrows()] 
        bboxes_centers = bboxes_centers_and_fill_outpoints(
            components,
            img_graph, 
            components_outpoints # filled by reference
        )

        debug_img = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        for component in components_outpoints:
            for outpoint in component:
                cv2.circle (debug_img, tuple(string_to_array(outpoint))[::-1], 8, (0,200,0), 8)

        image = self.__last_binarized.copy()
        for _, bbox_data in components.iterrows(): #erasing bboxes
            xmin, xmax, ymin, ymax = bbox_data[
                ['xmin', 'xmax', 'ymin', 'ymax']
            ].astype(int)
            p1, p2 = np.array([xmin+1, ymin+1]), np.array([xmax-1, ymax-1])
            cv2.rectangle (image, p1.astype(int), p2.astype(int), 0, -1)
        image = inflate(image, analyzed=self.__last_binarized)
        img_graph = ImageGraph(binarized_image=image)

        components['value'] = 0
        for index, data in components.iterrows(): # discovering and saving values
            components.at[index, 'value'] = self.nearest_value (data[POS_ATTR])

        #collisions = bboxes_collisions (components)
        #for collision in collisions:
        #    components = filter_collision(components, collision)

        # constructing outpoints graph
        vertices_number = 0
        outpoint_index: dict[str,int] = {}
        adjacency_list: list[set[int]] = []
        for outpoints_list in components_outpoints:
            for outpoint in outpoints_list:
                outpoint_index[outpoint] = vertices_number
                vertices_number += 1
                adjacency_list.append(set())
        vertex_node = [None] * vertices_number
    
        # ground nodes are always on node 0, setting it below
        grounds_index = []
        for component_index, component_data in components.iterrows(): 
            if component_data['name'].find ('ground') != -1:
                grounds_index.append(component_index)
                for out_index, outpoint in enumerate(
                        components_outpoints[component_index]
                    ):
                    vertex_node[outpoint_index[outpoint]] = 0

        # connect near outpoints of same component
        for component_index, component_data in components.iterrows(): 
            for out_index, outpoint in enumerate(
                    components_outpoints[component_index]
                ):
                nearest_outpoint = None
                shortest_angle = inf
                for other_outpoint in components_outpoints[component_index]:
                    if outpoint == other_outpoint:
                        continue
                    angle = angle_between(
                        string_to_array(outpoint) - bboxes_centers[component_index],
                        string_to_array(other_outpoint) - bboxes_centers[component_index]
                    )
                    if angle < shortest_angle:
                        shortest_angle = angle
                        nearest_outpoint = other_outpoint
                if nearest_outpoint is None: 
                    continue
                elif shortest_angle < MINIMUM_LINKING_NODE_ANGLE \
                and not (outpoint_index[outpoint] == outpoint_index[other_outpoint]):
                    #print (f'connecting a {outpoint_index[outpoint]} to {outpoint_index[other_outpoint]}')
                    adjacency_list[outpoint_index[outpoint]].add(outpoint_index[other_outpoint])
                    adjacency_list[outpoint_index[other_outpoint]].add(outpoint_index[outpoint])

        # connecting outpoints along the circuit
        for component_index, component_data in components.iterrows(): # for each component
            for out_index, outpoint in enumerate(
                    components_outpoints[component_index]
                ): # and each outpoint of that component
                if vertex_node[outpoint_index[outpoint]] is not None: 
                    continue

                for other_component_index, collision_point in \
                    img_graph.bfs_bbox_collisions( # ITERATING BY GENERATOR !
                        string_to_array(outpoint),
                        components
                    ): # so, for each outpoint connected to that one
                    # calculate nearest outpoint to the collision point
                    
                    min_distance = np.Infinity
                    nearest_outpoint = None
                    for sarray in components_outpoints[other_component_index]:
                        array = string_to_array(sarray)
                        distance = np.linalg.norm(collision_point - array)
                        if distance < min_distance:
                            nearest_outpoint = sarray
                            min_distance = distance
                    if nearest_outpoint == outpoint:
                        continue       
                    else: # connect the outpoint with the nearest point to collision
                        adjacency_list[outpoint_index[outpoint]].add(outpoint_index[nearest_outpoint])
                        adjacency_list[outpoint_index[nearest_outpoint]].add(outpoint_index[outpoint])

        index_to_point = {index: point for point, index in outpoint_index.items()}
        for vertex, neighbors in enumerate(adjacency_list):
            for neighbor in neighbors:
                cv2.line (
                    debug_img,
                    string_to_array(index_to_point[vertex])[::-1],
                    string_to_array(index_to_point[neighbor])[::-1],
                    (100,0,0),
                    10
                )

        print (outpoint_index)
        print (adjacency_list)
        plt.imshow (debug_img)
        plt.show()

         # removing grounds
        for i in grounds_index:
            components.drop(i, inplace=True)

        # propagating nodes along connected outpoints
        max_node = 1
        for vertex in range(vertices_number):
            #if (vertex_node[vertex] is not None):
            #    continue
            connected, lesser_node = bfs_anchieved_vertices_and_lesser_node(
                vertex,
                adjacency_list,
                vertex_node
            )
            for connected_vertex in connected:
                if lesser_node is None:
                    lesser_node = max_node
                    max_node += 1
                vertex_node[connected_vertex] = lesser_node

        #detecting terminals and linking to nodes
        components['anode'] = components['cathode'] = None
        for component_index, component_data in components.iterrows(): 
            if component_data['name'] in POLARIZED_COMPONENTS:
                #anode_point, cathode_point = self.polarization_points(component_data)
                # now search by proximity by a point to stole its node
                print ('WARNING: polarized component detected, but no terminals distinguishability')
            #else:
            anode, cathode = None, None
            for string_array in components_outpoints[component_index]:
                node = vertex_node[outpoint_index[string_array]]
                if anode is None:
                    anode = node
                elif node != anode:
                    cathode = node 
                if (anode and cathode) is not None:
                    break 
            if (anode and cathode) is None: # actually, works as "anode or cathode is None"
                continue 
            #end else

            components.at[component_index, 'anode'] = anode
            components.at[component_index, 'cathode'] = cathode
        
        print (vertex_node)

        # and finally you have the necessary to generate the netlist
        components_counts = {}
        netlist = ''
        for comp_index, comp_data in components.iterrows(): 
            letter = COMPONENT_LETTER[comp_data['name']]
            try:
                components_counts[letter] += 1
            except KeyError:
                components_counts[letter] = 1
            comp_data['schematic name'] = f'{letter}{components_counts[letter]}'
            netlist += f"{comp_data['schematic name']} " \
            + f"{comp_data['anode']} {comp_data['cathode']} {comp_data['value']}\n"

        return netlist

    def nearest_value(self, position:pandas.Series):
        return 0
        raise NotImplementedError()
    
    def polarization_points(self, data):
        raise NotImplementedError()