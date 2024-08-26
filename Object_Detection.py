import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Load class names
classes = []
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, img = cap.read()
    if not ret:
        break
    
    height, width, _ = img.shape

    # Create blob from image
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Get output layers and forward pass
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    # Process each detection
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Draw bounding boxes and labels
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)

    # Display the image
    cv2.imshow('Image', img)

    # Exit on 'ESC' key
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
