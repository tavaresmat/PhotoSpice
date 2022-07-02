from src.yolo_inference.component_detector import ComponentDetector
import cv2

IMAGE = 'test.png'
detector = ComponentDetector(weights='models/components mAP.97 close2x1 400ep.pt')
image = cv2.imread(IMAGE)
detection = detector.predict (image)
print (detection)
detector.show_binarized()
detector.plot()