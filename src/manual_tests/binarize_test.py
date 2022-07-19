import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.image_manipulation.utils import binarize


photo_path = f'test.jpeg'
image = cv2.imread(photo_path)
binarized = binarize(image)
plt.imshow (binarized, cmap='gray')
plt.axis('off')
plt.show()