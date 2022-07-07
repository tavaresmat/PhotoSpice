import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from src.image_manipulation.utils import binarize, inflate
from src.image_manipulation.image_graph import ImageGraph

IMAGEPATH = 'dataset/test/blobs.png'
image = binarize(cv2.imread(IMAGEPATH))
image = inflate(image)

image_graph = ImageGraph(binarized_image=image)
size0, size1 = image_graph.binarized_image.shape

plt.imshow(image)
plt.show()

for _ in image_graph.bfs_bbox_collisions(
    #np.array([200, 300]),
    np.array([380,238]),
    pd.DataFrame (
        {
            'xmin': [500, 280, 345, 240],
            'xmax': [550, 320, 370, 260],
            'ymin': [300, 100, 288, 365],
            'ymax': [350, 150, 306, 385]
        }
    )
): pass