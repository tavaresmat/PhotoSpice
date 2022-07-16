from matplotlib import pyplot as plt
from src.yolo_inference.component_detector import ComponentDetector
import cv2

from src.yolo_inference.netlist_generator import NetlistGenerator

IMAGES = [
    'test.jpeg',
    'test2.jpeg',
    'test3.jpeg'
]

netlist_generator = NetlistGenerator()
netlists = []
for imagepath in IMAGES:
    image = cv2.imread(imagepath)
    netlist = netlist_generator(image)
    netlists += [netlist]
    print (netlists[-1])
    netlist_generator.plot_debug_image()