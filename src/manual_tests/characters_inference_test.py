import cv2

from src.yolo_inference.character_box_detector import CharacterBoxDetector

IMAGE = 'test.jpeg'
detector = CharacterBoxDetector()
image = cv2.imread(IMAGE)
detection = detector.predict (image)
print (detection)
detector.plot()
print (detector.group_characters(image))
