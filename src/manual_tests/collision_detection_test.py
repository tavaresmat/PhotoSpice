import numpy as np
import pandas as pd

from src.augmentation.bbox_manipulation import plot_inference_bbox
from src.yolo_inference.node_detection_utils import bboxes_collisions


df = pd.DataFrame (
    [
        [50,150,0,100], 
        [900, 1000, 0, 100], 
        [900,1000,900,1000], 
        [0,100,900,1000], 
        [300, 700, 300, 700], 
        [50, 150, 850, 950],
        [550,950, 550, 950],
        [0, 200, 0, 200 ] # 7
    ],
    columns=['xmin','xmax','ymin','ymax']
)
df['name'] = 'any'
df['confidence'] = 0.99

print (bboxes_collisions(df))
plot_inference_bbox(np.zeros([1000,1000]), df)

#[[0, 7, 15000], [2, 6, 2500], [3, 5, 2500], [4, 6, 22500]]