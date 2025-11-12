import cv2
import numpy as np
import mss

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Set up screen capture - adjust these values for a smaller capture
sct = mss.mss()
monitor = {
    "top": 100,         # Starting from 100 pixels from the top
    "left": 100,        # Starting from 100 pixels from the left
    "width": 600,       # Width of the capture
    "height": 400       # Height of the capture
}

# Load class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

while True:
    # Capture the smaller portion of the screen
    img = sct.grab(monitor)
    img_np = np.array(img)
    frame = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)

    # Prepare the image for object detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]  # Scores for each class
            class_id = np.argmax(scores)  # Index of max score
            confidence = scores[class_id]  # Confidence level

            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

    cv2.imshow('Screen Capture with Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
