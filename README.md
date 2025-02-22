# Ramanth-
import cv2 import numpy as np import time from collections import defaultdict

Load YOLO Model

net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg") layer_names = net.getLayerNames() out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()] classes = []

with open("coco.names", "r") as f: classes = [line.strip() for line in f.readlines()]

Initialize video capture

cap = cv2.VideoCapture("traffic_video.mp4") fps = cap.get(cv2.CAP_PROP_FPS) pixel_to_meter_ratio = 0.04  # Adjust based on real-world mapping

Store previous positions

vehicle_tracks = defaultdict(list) speed_estimates = {}

while cap.isOpened(): ret, frame = cap.read() if not ret: break

height, width, _ = frame.shape
blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outputs = net.forward(out_layers)

class_ids = []
confidences = []
boxes = []

for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5 and classes[class_id] == "car":
            center_x, center_y, w, h = (detection[:4] * [width, height, width, height]).astype("int")
            x, y = int(center_x - w / 2), int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
current_frame_positions = {}

if len(indexes) > 0:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        center_x, center_y = x + w // 2, y + h // 2
        current_frame_positions[i] = (center_x, center_y)
        vehicle_tracks[i].append((center_x, center_y, time.time()))
        
        if len(vehicle_tracks[i]) > 2:
            x1, y1, t1 = vehicle_tracks[i][-2]
            x2, y2, t2 = vehicle_tracks[i][-1]
            distance_px = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            distance_m = distance_px * pixel_to_meter_ratio
            time_diff = t2 - t1
            if time_diff > 0:
                speed = (distance_m / time_diff) * 3.6  # Convert m/s to km/h
                speed_estimates[i] = speed
                
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{classes[class_ids[i]]}: {speed_estimates.get(i, 0):.2f} km/h"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

cv2.imshow("Vehicle Tracking", frame)
if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release() cv2.destroyAllWindows()

