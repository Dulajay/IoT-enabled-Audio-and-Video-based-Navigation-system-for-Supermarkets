from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

#video_path = "video.mp4"
# Open the camera feed
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the camera frames
while cap.isOpened():
    # Read a frame from the camera
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        if results is not None and len(results) > 0:  # Check if detections were made
            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = [box.id for box in results[0].boxes]  # Extract track IDs from the boxes

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=6)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if reading from the camera fails
        break

# Release the camera capture object and close the display window
cap.release()
cv2.destroyAllWindows()
