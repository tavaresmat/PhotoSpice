from src.yolo_inference.component_detector import ComponentDetector
import cv2

IMAGE = 'dataset/test/test2.png'
detector = ComponentDetector(weights='models/components mAP.97 close2x1 400ep.pt')
image = cv2.imread(IMAGE)
detection = detector.predict (image)
print (detection)
detector.plot()
netlist = detector.generate_netlist()
print (netlist)