import cv2
from ultralytics import YOLO
import torch
import numpy as np

# Check if CUDA (GPU) is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load your trained YOLOv8 model on the specified device
model = YOLO('best_model.pt').to(device)  # Replace 'best.pt' with the path to your model file

# Open the camera (0 for the default camera, or specify another index for external cameras)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Main loop to get predictions
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize the frame to be divisible by 32
    resized_frame = cv2.resize(frame, (640, 640))

    # Normalize the frame to [0, 1]
    normalized_frame = resized_frame / 255.0

    # Convert the frame to the correct format (BCHW) and move to the GPU
    tensor_frame = torch.from_numpy(normalized_frame).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # Get predictions
    results = model(tensor_frame)

    # Extract prediction details and draw bounding boxes
    annotated_frame = frame.copy()  # Make a copy of the original frame
    for result in results[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0].cpu().numpy()
        confidence = result.conf[0].cpu().numpy()
        class_id = int(result.cls[0].cpu().numpy())
        class_name = model.names[class_id]

        # Draw bounding box
        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # Draw label and confidence
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with annotations
    cv2.imshow('YOLOv8 Detection', annotated_frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
