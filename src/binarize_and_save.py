import os

import cv2

from utils import binarize

color_images_dir = "images/colored"
binary_images_dir = "images/binarized"
for file in os.listdir(color_images_dir):
    filepath = os.path.join(color_images_dir, file)
    image = cv2.imread(filepath)
    binary_image = binarize(image)
    cv2.imwrite (os.path.join(
            binary_images_dir,
            file
        ),
        binary_image
    )