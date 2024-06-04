import cv2
from ultralytics import YOLO
import csv

# Load the YOLOv8 model
model = YOLO("yolov8n-pose.pt")

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

# Define the codec and create a VideoWriter object to save the output video
output_path = "test1.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like 'XVID', 'DIVX', etc.
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Function to split the tensor data
def split_data(data):
    faces = []
    bodies = []
    legs = []

    for person in data:
        face = person[:4].tolist()
        body = person[4:11].tolist()
        leg = person[11:15].tolist()

        faces.append(face)
        bodies.append(body)
        legs.append(leg)

    return faces, bodies, legs

# Function to save data to a CSV file
def save_to_csv(filename, frame_data):
    headers = [
        'frame_number',
        'face_x0', 'face_y0', 'face_conf0', 'face_x1', 'face_y1', 'face_conf1', 'face_x2', 'face_y2', 'face_conf2', 'face_x3', 'face_y3', 'face_conf3', 'face_x4', 'face_y4', 'face_conf4',
        'body_x5', 'body_y5', 'body_conf5', 'body_x6', 'body_y6', 'body_conf6', 'body_x7', 'body_y7', 'body_conf7', 'body_x8', 'body_y8', 'body_conf8', 'body_x9', 'body_y9', 'body_conf9',
        'body_x10', 'body_y10', 'body_conf10', 'body_x11', 'body_y11', 'body_conf11',
        'leg_x12', 'leg_y12', 'leg_conf12', 'leg_x13', 'leg_y13', 'leg_conf13', 'leg_x14', 'leg_y14', 'leg_conf14', 'leg_x15', 'leg_y15', 'leg_conf15'
    ]

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for frame_number, faces, bodies, legs in frame_data:
            for face, body, leg in zip(faces, bodies, legs):
                flattened_row = [frame_number] + [item for sublist in face + body + leg for item in sublist]
                writer.writerow(flattened_row)

# Initialize frame number and data storage
frame_number = 0
frame_data = []

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, conf=0.5)

        # Visualize the results on the frame
        if results:
            annotated_frame = results[0].plot()
            # Extract keypoints data
            data = results[0].keypoints.data

            # Split the data into faces, bodies, and legs
            faces, bodies, legs = split_data(data)

            # Append the frame data for CSV output
            frame_data.append((frame_number, faces, bodies, legs))

            # Write the annotated frame to the output video file
            out.write(annotated_frame)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)
        else:
            print(f"Warning: No results found for frame {frame_number}.")

        # Increment the frame number
        frame_number += 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture and writer objects and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()

# Save the collected frame data to a CSV file
save_to_csv('test1.csv', frame_data)
