import cv2
import pandas as pd
import numpy as np
import os

# Load the CSV file
csv_path = 'test1.csv'
df = pd.read_csv(csv_path)

# Open the video file
video_path = "pose_sync_ive_baddie_1.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the video frame width, height, and frames per second (fps)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define colors for face, body, and legs
face_color = (255, 0, 0)  # Blue
body_color = (0, 255, 0)  # Green
leg_color = (0, 0, 255)  # Red

# Create output directory if it doesn't exist
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Function to overlay keypoints ㅂㅂㅂon the frame
def overlay_keypoints(frame, person):
    # Extract face, body, and leg keypoints
    face_x = person[['face_x0', 'face_x1', 'face_x2', 'face_x3', 'face_x4']].values
    face_y = person[['face_y0', 'face_y1', 'face_y2', 'face_y3', 'face_y4']].values
    body_x = person[['body_x5', 'body_x6', 'body_x7', 'body_x8', 'body_x9', 'body_x10', 'body_x11']].values
    body_y = person[['body_y5', 'body_y6', 'body_y7', 'body_y8', 'body_y9', 'body_y10', 'body_y11']].values
    leg_x = person[['leg_x12', 'leg_x13', 'leg_x14', 'leg_x15']].values
    leg_y = person[['leg_y12', 'leg_y13', 'leg_y14', 'leg_y15']].values

    # Draw keypoints and add text
    for i, (x, y) in enumerate(zip(face_x, face_y)):
        if not np.isnan(x) and not np.isnan(y) and x != 0 and y != 0:
            cv2.circle(frame, (int(x), int(y)), 5, face_color, -1)
            cv2.putText(frame, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 2)
    for i, (x, y) in enumerate(zip(body_x, body_y), start=5):
        if not np.isnan(x) and not np.isnan(y) and x != 0 and y != 0:
            cv2.circle(frame, (int(x), int(y)), 5, body_color, -1)
            cv2.putText(frame, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, body_color, 2)
    for i, (x, y) in enumerate(zip(leg_x, leg_y), start=12):
        if not np.isnan(x) and not np.isnan(y) and x != 0 and y != 0:
            cv2.circle(frame, (int(x), int(y)), 5, leg_color, -1)
            cv2.putText(frame, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, leg_color, 2)

    # Draw lines between keypoints
    if not np.isnan(face_x[0]) and not np.isnan(face_y[0]) and not np.isnan(face_x[1]) and not np.isnan(face_y[1]):
        cv2.line(frame, (int(face_x[0]), int(face_y[0])), (int(face_x[1]), int(face_y[1])), face_color, 2)
    if not np.isnan(body_x[0]) and not np.isnan(body_y[0]) and not np.isnan(body_x[1]) and not np.isnan(body_y[1]):
        cv2.line(frame, (int(body_x[0]), int(body_y[0])), (int(body_x[1]), int(body_y[1])), body_color, 2)
    if not np.isnan(leg_x[0]) and not np.isnan(leg_y[0]) and not np.isnan(leg_x[1]) and not np.isnan(leg_y[1]):
        cv2.line(frame, (int(leg_x[0]), int(leg_y[0])), (int(leg_x[1]), int(leg_y[1])), leg_color, 2)


# Initialize frame number
frame_number = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Get the pose data for the current frame
        frame_data = df[df['frame_number'] == frame_number]

        # Overlay keypoints for each person in the frame
        for idx in range(len(frame_data)):
            person = frame_data.iloc[idx]
            overlay_keypoints(frame, person)

        # Save the frame with keypoints
        output_path = os.path.join(output_dir, f"frame_{frame_number:04d}.jpg")
        cv2.imwrite(output_path, frame)

        # Display the frame with keypoints
        cv2.imshow("Pose Estimation", frame)

        # Increment the frame number
        frame_number += 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture and close the display window
cap.release()
cv2.destroyAllWindows()

