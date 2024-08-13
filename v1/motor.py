import cv2
import os
import serial
import time
from ultralytics import YOLO
import numpy as np
import torch

# Load YOLOv8 model (pretrained on COCO dataset) and move it to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = YOLO('yolov8s.pt').to(device)

# Define the class index for 'person' in COCO dataset
PERSON_CLASS_INDEX = 0

# Create directories for saving frames and annotations
os.makedirs('detected_frames', exist_ok=True)
os.makedirs('annotations', exist_ok=True)

# Initialize serial communication with Arduino
arduino = serial.Serial('COM3', 9600)
time.sleep(2)  # Wait for the serial connection to initialize

# Function to save annotations in YOLO format
def save_annotation(filename, boxes, img_width, img_height):
    with open(filename, 'w') as f:
        for box in boxes:
            x_center = (box[0] + box[2]) / 2 / img_width
            y_center = (box[1] + box[3]) / 2 / img_height
            width = (box[2] - box[0]) / img_width
            height = (box[3] - box[1]) / img_height
            f.write(f"{PERSON_CLASS_INDEX} {x_center} {y_center} {width} {height}\n")

# Function to check if one rectangle is completely inside another rectangle
def is_completely_inside(inner_rect, outer_rect):
    x1_min, y1_min, x1_max, y1_max = inner_rect
    x2_min, y2_min, x2_max, y2_max = outer_rect
    return (x1_min >= x2_min and y1_min >= y2_min and x1_max <= x2_max and y1_max <= y2_max)

# Function to check if a rectangle is outside another rectangle and determine direction
def get_direction_to_center(inner_rect, outer_rect):
    x1_min, y1_min, x1_max, y1_max = inner_rect
    x2_min, y2_min, x2_max, y2_max = outer_rect
    direction = ''
    if y1_min < y2_min:  # Person is above the center rectangle
        return 'S'  # Stop
    if x1_min < x2_min:
        direction += 'L'  # Move left
    elif x1_max > x2_max:
        direction += 'R'  # Move right
    if y1_max > y2_max:
        direction += 'D'  # Move down
    print(direction)
    return direction

# Open video capture (use 0 for webcam or replace with video file path)
cap = cv2.VideoCapture(0)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_height, img_width, _ = frame.shape

    # Adjust these variables to change the size of the center rectangle
    center_rect_width = img_width // 2  # Change this to adjust width
    center_rect_height = img_height // 1  # Change this to adjust height

    center_rect_x1 = (img_width - center_rect_width) // 2
    center_rect_y1 = (img_height - center_rect_height) // 2
    center_rect_x2 = center_rect_x1 + center_rect_width
    center_rect_y2 = center_rect_y1 + center_rect_height
    center_rect = (center_rect_x1, center_rect_y1, center_rect_x2, center_rect_y2)

    # Perform inference
    results = model(frame)

    # Process results
    boxes = []
    person_detected = False
    direction = ''
    for result in results:
        for pred in result.boxes:
            if int(pred.cls.item()) == PERSON_CLASS_INDEX:
                x1, y1, x2, y2 = map(int, pred.xyxy[0])
                person_rect = (x1, y1, x2, y2)
                
                person_width = x2 - x1
                if person_width > center_rect_width:
                    direction = 'S'  # Send stop command if person width is greater
                    arduino.write(b'S')
                    break
                
                if is_completely_inside(person_rect, center_rect):
                    person_detected = True
                    boxes.append([x1, y1, x2, y2])

                    # Draw bounding box on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    direction = get_direction_to_center(person_rect, center_rect)

    # Draw center rectangle on the frame for display
    display_frame = frame.copy()
    cv2.rectangle(display_frame, (center_rect_x1, center_rect_y1), (center_rect_x2, center_rect_y2), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Camera Frame', display_frame)

    # Send command to Arduino if direction is not 'S'
    if direction != 'S':
        if person_detected:
            arduino.write(b'F')  # Send 'F' for forward
        else:
            if direction == 'L':
                arduino.write(b'L')  # Send 'L' for left
            elif direction == 'R':
                arduino.write(b'R')  # Send 'R' for right
            else:
                arduino.write(b'S')  # Send 'S' for stop

    # Save the frame and annotation only if a person is completely inside the center rectangle
    if person_detected:
        # Save the frame with bounding boxes
        #frame_filename = f'detected_frames/frame_{frame_count:04d}.jpg'
        #cv2.imwrite(frame_filename, frame)

        # Save the annotation
        #annotation_filename = f'annotations/frame_{frame_count:04d}.txt'
        #save_annotation(annotation_filename, boxes, frame.shape[1], frame.shape[0])

        frame_count += 1

    # Exit feature: Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
