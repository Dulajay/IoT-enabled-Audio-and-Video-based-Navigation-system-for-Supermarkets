import cv2
import os
import serial
import time
from ultralytics import YOLO
import numpy as np
import torch
import signal
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_pose = YOLO('yolov8n-pose.pt').to(device)  # Load pose model
model_track = YOLO('yolov8s.pt').to(device)  # Load tracking model

PERSON_CLASS_INDEX = 0

os.makedirs('detected_frames', exist_ok=True)
os.makedirs('annotations', exist_ok=True)

arduino = serial.Serial('COM10', 9600)
time.sleep(2)  

def save_annotation(filename, boxes, img_width, img_height):
    with open(filename, 'w') as f:
        for box in boxes:
            x_center = (box[0] + box[2]) / 2 / img_width
            y_center = (box[1] + box[3]) / 2 / img_height
            width = (box[2] - box[0]) / img_width
            height = (box[3] - box[1]) / img_height
            f.write(f"{PERSON_CLASS_INDEX} {x_center} {y_center} {width} {height}\n")

def is_completely_inside(inner_rect, outer_rect):
    x1_min, y1_min, x1_max, y1_max = inner_rect
    x2_min, y2_min, x2_max, y2_max = outer_rect
    return (x1_min >= x2_min and y1_min >= y2_min and x1_max <= x2_max and y1_max <= y2_max)

def get_direction_to_center(inner_rect, outer_rect):
    x1_min, y1_min, x1_max, y1_max = inner_rect
    x2_min, y2_min, x2_max, y2_max = outer_rect
    direction = ''
    if y1_min < y2_min:  
        return 'S'  
    if x1_min < x2_min:
        direction += 'L'  
    elif x1_max > x2_max:
        direction += 'R'  
    if y1_max > y2_max:
        direction += 'D'  
    print(direction)
    return direction

cap = cv2.VideoCapture(0)

frame_count = 0

def signal_handler(sig, frame):
    print("Stopping motor...")
    arduino.write(b'S') 
    arduino.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Person tracking dictionary
person_tracker = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_height, img_width, _ = frame.shape

    center_rect_width = img_width // 2  
    center_rect_height = img_height // 1  

    center_rect_x1 = (img_width - center_rect_width) // 2
    center_rect_y1 = (img_height - center_rect_height) // 2
    center_rect_x2 = center_rect_x1 + center_rect_width
    center_rect_y2 = center_rect_y1 + center_rect_height
    center_rect = (center_rect_x1, center_rect_y1, center_rect_x2, center_rect_y2)

    # Perform inference and tracking
    results = model_track.track(frame, persist=500, tracker="bytetrack.yaml")

    boxes = []
    person_detected = False
    direction = ''
    
    for result in results:
        for pred in result.boxes:
            if int(pred.cls.item()) == PERSON_CLASS_INDEX:
                x1, y1, x2, y2 = map(int, pred.xyxy[0])
                person_rect = (x1, y1, x2, y2)
                person_width = x2 - x1

                if pred.id is not None:
                    person_id = int(pred.id.item())

                    # Display the bounding box and ID for each person
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {person_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    if person_id == 1:
                        print(f"Person ID {person_id} detected, giving motor commands")
                        if person_width > center_rect_width:
                            direction = 'F'  # Stop if person width is greater
                            arduino.write(b'F')
                        elif is_completely_inside(person_rect, center_rect):
                            person_detected = True
                            boxes.append([x1, y1, x2, y2])
                        else:
                            direction = get_direction_to_center(person_rect, center_rect)

    # Perform pose detection
    pose_results = model_pose(frame)

    for pose_result in pose_results:
        if hasattr(pose_result, 'keypoints') and pose_result.keypoints is not None:
            if pose_result.keypoints.conf is not None:
                keypoints = pose_result.keypoints.xy[0].cpu().numpy()  # Get the (x, y) coordinates
                confidences = pose_result.keypoints.conf[0].cpu().numpy()  # Get confidence values

                for i, (kp, conf) in enumerate(zip(keypoints, confidences)):
                    x_kp, y_kp = int(kp[0]), int(kp[1])
                    if conf > 0.5:
                        cv2.circle(frame, (x_kp, y_kp), 5, (0, 0, 255), -1)  # Draw keypoint
                        cv2.putText(frame, f"{i}", (x_kp, y_kp - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            else:
                print("No confidence values detected for keypoints.")
        else:
            print("No keypoints detected in this frame.")

    display_frame = frame.copy()

    # Draw the center rectangle on the frame for reference
    cv2.rectangle(display_frame, (center_rect_x1, center_rect_y1), (center_rect_x2, center_rect_y2), (255, 0, 0), 2)

    cv2.imshow('Camera Frame', display_frame)

    # Send command to Arduino if direction is not 'S'
    if direction != 'S':
        if person_detected:
            arduino.write(b'F')  
        else:
            if direction == 'L':
                arduino.write(b'L')  
            elif direction == 'R':
                arduino.write(b'R')  
            else:
                arduino.write(b'S')  

    if person_detected:
        frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        arduino.write(b'S')
        break

cap.release()
cv2.destroyAllWindows()
