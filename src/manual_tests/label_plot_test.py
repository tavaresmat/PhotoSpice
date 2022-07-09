from src.augmentation.bbox_manipulation import plot_bbox_from_path

img = 'samplei3.var3'
imgext = 'png'
_set = 'validation_aug'
imagepath = f'other_datasets/numbers2/{_set}/images/{img}.{imgext}' # image that will be tested for augmentation
bboxespath = f'other_datasets/numbers2/{_set}/labels/{img}.txt' # bboxes file in yolo format

plot_bbox_from_path (
    imagepath,
    bboxespath
)