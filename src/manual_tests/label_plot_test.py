from src.augmentation.bbox_manipulation import plot_bbox_from_path

img = 'samplei3.var3'
imgext = 'png'
_set = 'validation_aug'
imagepath = f'teste charac/characters_bin/0-055.e3.png' # image that will be tested for augmentation
bboxespath = f'teste charac/labels/0-055.e3.txt' # bboxes file in yolo format

plot_bbox_from_path (
    imagepath,
    bboxespath
)