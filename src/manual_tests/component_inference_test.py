from src.yolo_inference.component_detector import ComponentDetector
import cv2

IMAGE = 'drawing.png'
detector = ComponentDetector(weights='models/best.pt')
image = cv2.imread(IMAGE)
detection = detector.predict (image)
print (detection)
detector.show_binarized()
detector.plot()