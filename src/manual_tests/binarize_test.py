import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.image_manipulation.utils import binarize

samples = None
final = None
for i in range (0,3): 
    photo_path = f'dataset/cimages/sampleA{i}.jpg'
    image = cv2.imread(photo_path)
    binarized = cv2.resize(binarize(image), (1000, 1000))
    resized = cv2.resize (image, (1000,1000))
    samples = resized if samples is None else np.concatenate([samples, resized], axis=1)
    final = binarized if final is None else np.c_[final, binarized]

samples = cv2.cvtColor(samples, cv2.COLOR_BGR2RGB)
final =  cv2.cvtColor(final, cv2.COLOR_GRAY2RGB)
plt.imshow (np.concatenate([final, samples]))
plt.axis('off')
plt.show()