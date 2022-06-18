from src.yolo_inference.component_detector import ComponentDetector
import cv2

IMAGE = 'test2.jpg'
detector = ComponentDetector(weights='models/components mAP.93. close2x1.pt')
image = cv2.imread(IMAGE)
detection = detector.predict_binarized (image)
print (detection)
detector.plot()