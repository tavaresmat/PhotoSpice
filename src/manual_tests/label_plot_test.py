from src.augmentation.bbox_manipulation import plot_bbox_from_path

img = 'img1'
imgext = 'png'
_set = 'train'
dataset_name = "numbers2"
imagepath = f'other_datasets/{dataset_name}/{_set}/images/{img}.{imgext}' # image that will be tested for augmentation
bboxespath = f'other_datasets/{dataset_name}/{_set}/labels/{img}.txt' # bboxes file in yolo format

plot_bbox_from_path (
    imagepath,
    bboxespath
)
