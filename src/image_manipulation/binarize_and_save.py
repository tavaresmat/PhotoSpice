import os

import cv2

from src.image_manipulation.utils import binarize

COLOR_SOURCE_DIR = "other_datasets/numbers/old_images"
BINARY_DESTINATION_DIR = "test bin"

if __name__ == '__main__':    
    for file in os.listdir(COLOR_SOURCE_DIR):
        filepath = os.path.join(COLOR_SOURCE_DIR, file)
        image = cv2.imread(filepath)
        binary_image = binarize(image)
        cv2.imwrite (os.path.join(
                BINARY_DESTINATION_DIR,
                file
            ),
            binary_image
        )