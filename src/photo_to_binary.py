import cv2
import numpy as np
import matplotlib.pyplot as plt

def binarize (image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5,5), 1)
    threshold_image = cv2.Canny (blurred_image, 30, 70) 
    kernel = np.ones ((2, 2))
    closed_threshold = cv2.morphologyEx (threshold_image, cv2.MORPH_CLOSE, kernel, iterations=3)
    return closed_threshold


samples = None
final = None
for i in range (1,6): 
    photo_path = f'images/sample{i}.jpg'
    image = cv2.imread(photo_path)
    binarized = cv2.resize(binarize (image), (1000, 1000))
    resized = cv2.resize (image, (1000,1000))
    samples = resized if samples is None else np.concatenate([samples, resized], axis=1)
    final = binarized if final is None else np.c_[final, binarized]

samples = cv2.cvtColor(samples, cv2.COLOR_BGR2RGB)
final =  cv2.cvtColor(final, cv2.COLOR_GRAY2RGB)
plt.imshow (np.concatenate([final, samples]))
plt.axis('off')
plt.show()