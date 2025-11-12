import cv2

try:
    net = cv2.dnn.readNet("C:/Users/Uporabnik/Downloads/Snap/yolov3.weights", "C:/Users/Uporabnik/Downloads/Snap/yolov3.cfg")
    print("YOLO model loaded successfully!")
except cv2.error as e:
    print(f"Error loading YOLO model: {e}")
