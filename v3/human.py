import cv2
import os
import serial
import time
from ultralytics import YOLO
import numpy as np
import torch
from collections import deque
from torch.cuda.amp import autocast, GradScaler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


model = YOLO('yolov8s.pt').to(device)


PERSON_CLASS_INDEX = 0


os.makedirs('detected_frames', exist_ok=True)
os.makedirs('annotations', exist_ok=True)


arduino = serial.Serial('COM3', 9600)  
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
    return direction


cap = cv2.VideoCapture(0)

fps_history = deque(maxlen=30)
prev_time = time.time()
scaler = GradScaler()

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    elapsed_time = current_time - prev_time
    prev_time = current_time

    fps = 1.0 / elapsed_time
    fps_history.append(fps)
    avg_fps = sum(fps_history) / len(fps_history)

    img_height, img_width, _ = frame.shape

    
    center_rect_width = img_width // 2  # Change this to adjust width
    center_rect_height = img_height // 1  # Change this to adjust height

    center_rect_x1 = (img_width - center_rect_width) // 2
    center_rect_y1 = (img_height - center_rect_height) // 2
    center_rect_x2 = center_rect_x1 + center_rect_width
    center_rect_y2 = center_rect_y1 + center_rect_height
    center_rect = (center_rect_x1, center_rect_y1, center_rect_x2, center_rect_y2)

    # Resize the frame to a compatible size
    resized_frame = cv2.resize(frame, (640, 640))

    # Convert frame to tensor, normalize, add batch dimension, and move to GPU
    frame_tensor = torch.from_numpy(resized_frame).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    # Perform inference with mixed precision
    with autocast():
        results = model(frame_tensor)

    # Process results
    boxes = []
    person_detected = False
    direction = ''
    for result in results:
        for pred in result.boxes:
            if int(pred.cls.item()) == PERSON_CLASS_INDEX:
                x1, y1, x2, y2 = map(int, pred.xyxy[0])
                person_rect = (x1, y1, x2, y2)
                boxes.append([x1, y1, x2, y2])
                if is_completely_inside(person_rect, center_rect):
                    person_detected = True

                   
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    direction = get_direction_to_center(person_rect, center_rect)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    
    display_frame = frame.copy()
    cv2.rectangle(display_frame, (center_rect_x1, center_rect_y1), (center_rect_x2, center_rect_y2), (255, 0, 0), 2)

    cv2.putText(display_frame, f"FPS: {avg_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    
    cv2.imshow('Camera Frame', display_frame)

    
    if person_detected:
        arduino.write(b'F')  
    else:
        if direction == 'L':
            arduino.write(b'L') 
        elif direction == 'R':
            arduino.write(b'R') 
        elif direction == 'U':
            arduino.write(b'U') 
        elif direction == 'D':
            arduino.write(b'D')  
        else:
            arduino.write(b'S')  

    
    if boxes:
        frame_filename = f'detected_frames/frame_{frame_count:04d}.jpg'
        cv2.imwrite(frame_filename, frame)

        annotation_filename = f'annotations/frame_{frame_count:04d}.txt'
        save_annotation(annotation_filename, boxes, frame.shape[1], frame.shape[0])

        frame_count += 1

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
