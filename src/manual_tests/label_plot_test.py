from src.augmentation.bbox_manipulation import plot_bbox_from_path

img = 'sampleL4b.var1'
imgext = 'jpg'
_set = 'aug_train'
imagepath = f'dataset/{_set}/images/{img}.{imgext}' # image that will be tested for augmentation
bboxespath = f'dataset/{_set}/labels/{img}.txt' # bboxes file in yolo format

plot_bbox_from_path (
    imagepath,
    bboxespath
)