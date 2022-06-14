import torch
import cv2

model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best 400ep map.91.pt', device='cpu')

img = cv2.imread('dataset/bimages/sampleA0.jpg')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
# Inference
results = model(img, size=600)  # includes NMS

# Results
results.print()  
#results.show()  # or .show()

#results = results.xyxy[0]  # img1 predictions (tensor)
boxes = results.pandas().xyxy[0]  # img1 predictions (pandas)
print (boxes)