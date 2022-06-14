import torch
import cv2
from src.augmentation.bbox_manipulation import plot_inference_bbox



model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best 400ep map.91.pt', device='cpu')

img = cv2.imread('dataset/bimages/sampleD1.jpeg')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
# Inference
results = model(img, size=600)  # includes NMS

# Results
#results.show()  # or .show()
boxes = results.pandas().xyxy[0]
#results = results.xyxy[0]  # img1 predictions (tensor)
#boxes = boxes[list(boxes.columns)[:-2] + list(boxes.columns)[-1:]]  # img1 predictions (pandas)

plot_inference_bbox(img, boxes, 1, 4)