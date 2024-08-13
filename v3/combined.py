import cv2
import os
import serial
import time
from ultralytics import YOLO
import numpy as np
import torch
import signal
import sys
import threading

# Setup device for YOLO model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = YOLO('yolov8s.pt').to(device)

PERSON_CLASS_INDEX = 0

os.makedirs('detected_frames', exist_ok=True)
os.makedirs('annotations', exist_ok=True)

# Serial connections
arduino_motor = serial.Serial('COM3', 9600)
arduino_lidar = serial.Serial('COM6', 115200)
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

# Lidar Data class for obstacle detection
class LidarData():
    def __init__(self, serial_connection):
        self.serial_connection = serial_connection
        self.DATA_LENGTH = 7
        self.MAX_DISTANCE = 3000
        self.MIN_DISTANCE = 100
        self.MAX_DATA_SIZE = 360
        self.obstacle_detected = False

    def update_data(self):
        while True:
            try:
                if self.serial_connection.in_waiting > 0:
                    line = self.serial_connection.readline().decode().rstrip()
                    sensorData = line.split('\t')
                    if len(sensorData) == self.DATA_LENGTH:
                        for i in range(2, 6):
                            try:
                                dist = float(sensorData[i])
                                if self.MIN_DISTANCE <= dist <= self.MAX_DISTANCE:
                                    self.obstacle_detected = True
                                    return
                            except:
                                continue
                    self.obstacle_detected = False
            except KeyboardInterrupt:
                sys.exit()

lidar_data = LidarData(arduino_lidar)

def run_lidar():
    lidar_data.update_data()

# Start the lidar thread
lidar_thread = threading.Thread(target=run_lidar)
lidar_thread.daemon = True
lidar_thread.start()

cap = cv2.VideoCapture(0)

def signal_handler(sig, frame):
    print("Stopping motors...")
    arduino_motor.write(b'S')
    arduino_motor.close()
    arduino_lidar.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

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

    # Perform inference
    results = model(frame)
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
                    direction = 'S'
                    arduino_motor.write(b'S')
                    break
                if is_completely_inside(person_rect, center_rect):
                    person_detected = True
                    boxes.append([x1, y1, x2, y2])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    direction = get_direction_to_center(person_rect, center_rect)

    display_frame = frame.copy()
    cv2.rectangle(display_frame, (center_rect_x1, center_rect_y1), (center_rect_x2, center_rect_y2), (255, 0, 0), 2)
    cv2.imshow('Camera Frame', display_frame)

    if lidar_data.obstacle_detected:
        arduino_motor.write(b'S')
    elif direction != 'S':
        if person_detected:
            arduino_motor.write(b'F')
        else:
            if direction == 'L':
                arduino_motor.write(b'L')
            elif direction == 'R':
                arduino_motor.write(b'R')
            else:
                arduino_motor.write(b'S')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
