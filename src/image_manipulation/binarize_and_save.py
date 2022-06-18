import os

import cv2

from src.image_manipulation.utils import binarize

if __name__ == '__main__':
    color_images_dir = "dataset/cimages"
    binary_images_dir = "dataset/bimages"
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