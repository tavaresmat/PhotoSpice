from math import inf
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas
from src.augmentation.bbox_manipulation import draw_inference_bbox

from src.image_manipulation.image_graph import ImageGraph
from src.image_manipulation.utils import downgrade_image, inflate
from src.yolo_inference.component_detector import ComponentDetector
from src.yolo_inference.node_detection_utils import (
    angle_between,
    bboxes_centers_and_fill_outpoints,
    bfs_anchieved_vertices_and_lesser_node,
    string_to_array,
    filter_bboxes_overlap_by_confidence
)

COMPONENTS_MIN_CONFIDENCE = 0.40
MAX_OVERLAP_AREA = 0.50
MAX_INPUT_PIXELS = 250_000
MINIMUM_LINKING_NODE_ANGLE = 80

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

class NetlistGenerator:

    components:pandas.DataFrame = None
    debug_image: np.ndarray = None
    component_detector = None

    def __init__(self):
        self.component_detector = ComponentDetector()

    def plot_debug_image(self) -> None:
        plt.imshow (self.debug_image)
        plt.show()

    def __call__(self, image:np.ndarray) -> str:

        self.debug_image = None
        image = downgrade_image(image, MAX_INPUT_PIXELS)
        self.components = self.component_detector.predict(image) # self.components list
        image = self.component_detector.last_binarized
        image, _ = inflate(image)
        img_graph = ImageGraph(binarized_image=image)
        
        self.remove_low_confidences()
        collisions = filter_bboxes_overlap_by_confidence (
            self.components,
            MAX_OVERLAP_AREA
        )

        components_outpoints = [[] for _,_ in self.components.iterrows()] 
        bboxes_centers = bboxes_centers_and_fill_outpoints(
            self.components,
            img_graph, 
            components_outpoints # filled by reference
        )

        self.debug_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        for component in components_outpoints:
            for outpoint in component:
                cv2.circle (self.debug_image, tuple(string_to_array(outpoint))[::-1],
                 self.debug_image.shape[0]//100, (0,200,0), 8)

        image = self.component_detector.last_binarized.copy()
        for _, bbox_data in self.components.iterrows(): #erasing bboxes
            xmin, xmax, ymin, ymax = bbox_data[
                ['xmin', 'xmax', 'ymin', 'ymax']
            ].astype(int)
            p1, p2 = np.array([xmin+1, ymin+1]), np.array([xmax-1, ymax-1])
            new_p1 = .9*p1 + .1*p2
            new_p2 = .9*p2 + .1*p1
            cv2.rectangle (image, new_p1.astype(int), new_p2.astype(int), 0, -1)
        image, trace_median_width = inflate(image, analyzed=self.component_detector.last_binarized)
        img_graph = ImageGraph(binarized_image=image, grid=trace_median_width*0.5)

        self.components['value'] = 0
        for index, data in self.components.iterrows(): # discovering and saving values
            self.components.at[index, 'value'] = self.nearest_value (data[POS_ATTR])

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
        for component_index, component_data in self.components.iterrows(): 
            if component_data['name'].find ('ground') != -1:
                grounds_index.append(component_index)
                for out_index, outpoint in enumerate(
                        components_outpoints[component_index]
                    ):
                    vertex_node[outpoint_index[outpoint]] = 0

        # connect near outpoints of same component
        for component_index, component_data in self.components.iterrows(): 
            for out_index, outpoint in enumerate(
                    components_outpoints[component_index]
                ):

                nearest_outpoint = None
                shortest_angle = inf
                for other_outpoint in components_outpoints[component_index]:
                    if outpoint == other_outpoint:
                        continue
                    p1 = string_to_array(outpoint)
                    p2 = string_to_array(other_outpoint)
                    center = bboxes_centers[component_index].astype(int)
                    angle = angle_between(
                        p1 - center,
                        p2 - center
                    )

                    if angle < shortest_angle:
                        shortest_angle = angle
                        nearest_outpoint = other_outpoint
                if nearest_outpoint is None: 
                    continue

                elif shortest_angle < MINIMUM_LINKING_NODE_ANGLE :
                    adjacency_list[outpoint_index[outpoint]].add(outpoint_index[nearest_outpoint])
                    adjacency_list[outpoint_index[nearest_outpoint]].add(outpoint_index[outpoint])

        # connecting outpoints along the circuit
        for component_index, component_data in self.components.iterrows(): # for each component
            print (f'searching connections: component {component_index}')
            for out_index, outpoint in enumerate(
                    components_outpoints[component_index]
                ): # and each outpoint of that component
                if vertex_node[outpoint_index[outpoint]] is not None: 
                    continue

                for other_component_index, collision_point in \
                    img_graph.bfs_bbox_collisions( # ITERATING BY GENERATOR !
                        string_to_array(outpoint),
                        self.components
                    ): # so, for each outpoint connected to that one
                    # calculate nearest outpoint to the collision point
                    #print (f'{component_index}-{out_index}-{other_component_index}')
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
                        #print (f'{outpoint} <-> {nearest_outpoint}')
                        adjacency_list[outpoint_index[outpoint]].add(outpoint_index[nearest_outpoint])
                        adjacency_list[outpoint_index[nearest_outpoint]].add(outpoint_index[outpoint])

        index_to_point = {index: point for point, index in outpoint_index.items()}

        for vertex, neighbors in enumerate(adjacency_list):
            for neighbor in neighbors:
                cv2.line (
                    self.debug_image,
                    string_to_array(index_to_point[vertex])[::-1],
                    string_to_array(index_to_point[neighbor])[::-1],
                    (255,0,0),
                    self.debug_image.shape[0]//100
                )
        '''print (outpoint_index)
        print (adjacency_list)'''
        draw_inference_bbox(self.debug_image, self.components)

         # removing grounds
        for i in grounds_index:
            self.components.drop(i, inplace=True)

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
        self.components['anode'] = self.components['cathode'] = None
        for component_index, component_data in self.components.iterrows(): 
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
            if anode is None: #not connected to anything
                continue
            elif cathode is None: # TERMINAL UNCONNECTED
                if len(components_outpoints[component_index]) < 2:
                    continue
                else: # COMPONENT IN SHORT-CIRCUIT, DONT FORGET TO DECIDE WHAT TO DO
                    cathode = anode

            self.components.at[component_index, 'anode'] = anode
            self.components.at[component_index, 'cathode'] = cathode

        #choose a ground
        # and finally you have the necessary to generate the netlist
        components_counts = {}
        netlist = ''
        for comp_index, comp_data in self.components.iterrows(): 
            letter = COMPONENT_LETTER[comp_data['name']]
            try:
                components_counts[letter] += 1
            except KeyError:
                components_counts[letter] = 1

            comp_data['schematic name'] = f'{letter}{components_counts[letter]}'
            name = comp_data['schematic name']
            anode = comp_data['anode']
            cathode = comp_data['cathode']
            value = comp_data['value']
            if (comp_data['anode'] is None) or (comp_data['cathode'] is None):
                continue

            netlist += f"{name} {anode} {cathode} {value}\n"

        return netlist

    def remove_low_confidences(self):
        to_drop = []
        for component_index, component_data \
        in self.components.iterrows(): # for each component
            if (component_data['confidence'] < COMPONENTS_MIN_CONFIDENCE): 
                to_drop.append(component_index)
        self.components.drop(to_drop, axis=0, inplace=True)
        self.components.reset_index(inplace=True)

    def nearest_value(self, position:pandas.Series):
        return 0
        raise NotImplementedError()
    
    def polarization_points(self, data):
        raise NotImplementedError()