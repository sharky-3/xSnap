import cv2
import numpy as np
import mss
import pyautogui  # Import pyautogui to move the mouse

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Set up screen capture
sct = mss.mss()

# Get the full screen resolution
monitor = sct.monitors[1]  # Assuming the main monitor is the first one (index 1)
screen_width = monitor["width"]
screen_height = monitor["height"]

# Define the capture region (centered 1280x720 area)
capture_width = 1280
capture_height = 720

# Calculate the top-left corner to center the capture region
top = (screen_height - capture_height) // 2
left = (screen_width - capture_width) // 2

# Set the capture area
monitor_capture = {
    "top": top,
    "left": left,
    "width": capture_width,
    "height": capture_height
}

# Load class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define a flag to lock the cursor
lock_cursor = False  # Set to False to unlock the cursor

while True:
    # Capture the specified portion of the screen (centered region)
    img = sct.grab(monitor_capture)
    img_np = np.array(img)
    frame = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)

    # Prepare the image for object detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
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

    # Track the center of the largest detected object (if any)
    mouse_x = None
    mouse_y = None

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

            # Calculate the center of the bounding box
            box_center_x = x + w // 2
            box_center_y = y + h // 2

            # If the cursor is locked, move it to the center of the detected object
            if lock_cursor:
                mouse_x = box_center_x
                mouse_y = box_center_y

    # If cursor locking is enabled and an object was detected, move the mouse
    if lock_cursor and mouse_x is not None and mouse_y is not None:
        pyautogui.moveTo(left + mouse_x, top + mouse_y)

    # Show the captured frame with object detection
    cv2.imshow('Screen Capture with Object Detection', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
