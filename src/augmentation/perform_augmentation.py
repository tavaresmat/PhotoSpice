import os
import shutil
from random import shuffle
from tempfile import TemporaryDirectory
from turtle import clear

import matplotlib.pyplot as plt
import numpy as np
import cv2
import albumentations as alb

from src.augmentation.bbox_manipulation import (
    load_bbox_list,
    save_bbox_list
)
from src.image_manipulation.utils import open_rgb_image

GENERATED_IMAGES_SIZE = {'height':600, 'width':800}

just_resize = alb.Compose ([
            alb.augmentations.geometric.resize.Resize (**GENERATED_IMAGES_SIZE, always_apply=True)
        ],
        bbox_params=alb.BboxParams(format='yolo', min_visibility=0.82)
    )

augmentation_pipeline = alb.Compose (
        [
            alb.augmentations.geometric.resize.Resize (**GENERATED_IMAGES_SIZE, always_apply=True),
            #alb.augmentations.geometric.transforms.Perspective (scale=(0.04, 0.09)),
            alb.augmentations.geometric.transforms.Perspective (scale=(0.02, 0.05)),
            #alb.augmentations.transforms.Flip(p=0.6),
            alb.augmentations.geometric.transforms.Affine (
                scale=(0.85,1), translate_percent=(-0.02, 0.02),
                rotate=(-10,10), shear={'x':(-15,15) , 'y': (-15,15)},
                cval=0, always_apply=True
            ),
            alb.augmentations.transforms.GaussNoise (
                var_limit=60, mean=0, per_channel=False, always_apply=True
            ),
        ],
        bbox_params=alb.BboxParams(format='yolo', min_visibility=0.7)
)

def augmentation_on_dataset (source_dir:str , destination_dir:str , aug_scale:int = 6, division:list[float]=[1.0]):
    """
    Performs data augmentation on the directory indicated

    args:
        source_dir (str): path to a directory, that must have 
            "images" and "labels" subdirectories, where "images"
            contains the images of dataset, and "labels" contains
            txt files, with same name of those on images 
            (excepting by pos-dot terminating). 
            These txt files are the bouding boxes file, in YOLO 
            format.

        destination_dir (str): path to destination directory, that may be empty

        aug_scale (int): how many new images each original source file will derive by
            data augmentation process (default in 6)

        division (tuple of floats): if the final dataset must be divided 
            (such as in train, test and validation), it tells the proportions
            of each division. Total sum must be equals to 1.0
    """
    assert aug_scale > 0
    assert sum(division) == 1.0
    img_souce_dir = os.path.join (source_dir, 'images')
    labels_source_dir = os.path.join (source_dir, 'labels')
    assert os.path.exists (labels_source_dir)
    assert os.path.exists (img_souce_dir)

    splitting = (len(division) > 1)
    for i in range(len(division)):
        destination_paths = [
            os.path.join (destination_dir, f'{i}/images' if splitting else 'images'),
            os.path.join (destination_dir, f'{i}/labels' if splitting else 'labels'),
        ]
        for path in destination_paths:
            if not os.path.exists(path):
                os.makedirs(path)
                print(f'The new directory "{path}" was created')

    new_files: "list[tuple]" = [] # newly created images and text files

    with TemporaryDirectory() as tempdirname:
        for imagename, labelsname in zip( 
            sorted(os.listdir(img_souce_dir)),
            sorted(os.listdir (labels_source_dir))
        ):
            imagepath = os.path.join(img_souce_dir, imagename)
            labelspath = os.path.join(labels_source_dir, labelsname)
            image = open_rgb_image(imagepath)
            bboxes = load_bbox_list (labelspath)
            dot_pos = imagename.find('.', -5)
            new_name = ""
            for i in range (aug_scale):
                if i != 0:          
                    transformed = augmentation_pipeline (image=image, bboxes=bboxes)
                else: # no augmentation on i == 0
                    transformed = just_resize (image=image, bboxes=bboxes)

                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']
                new_name = imagename[:dot_pos] + f'.var{i}' 
                new_imagepath = os.path.join(tempdirname, new_name + imagename[dot_pos:])
                new_labelspath = os.path.join (tempdirname, new_name + '.txt')
                plt.imsave (new_imagepath, transformed_image)
                save_bbox_list (new_labelspath, transformed_bboxes)
                new_files.append((new_imagepath, new_labelspath, new_name + imagename[dot_pos:]))
            print (f'augmented "{new_name}" successfully')

        shuffle (new_files)
        total_new_files = len(new_files)
        moving_file = 0
        for i in range(len(division)):
            moved_now = 0
            while (moved_now < round(total_new_files * division[i])):
                imgpath, bboxespath, imgname = new_files[moving_file]
                shutil.move(imgpath,
                    os.path.join (
                        destination_dir,
                        f'{i}/images' if splitting else f'images',
                        imgname
                    )
                )
                shutil.move(bboxespath,
                    os.path.join (
                        destination_dir,
                        f'{i}/labels' if splitting else f'labels',
                        imgname[:imgname.find('.', -5)] + '.txt'
                    )
                )
                moved_now += 1
                moving_file += 1
                print (f'"{imgname}" derived images written into final destination successfully')

if __name__ == "__main__":
    augmentation_on_dataset (
        'other_datasets/numbers2/validation', 
        'other_datasets/numbers2/validation_aug', 
        aug_scale=4
    )
