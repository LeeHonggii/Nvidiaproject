import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Setup scaling and face detector/predictor
scaler = 1
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture("video/aespa_forever.mp4")

# Check if video file can be opened
if not cap.isOpened():
    print("Error: Video file could not be opened.")
    exit(1)

ret, frame = cap.read()
if not ret:
    print("Error: No frames to read.")
    exit(1)

height, width, _ = frame.shape
fig_width = 10  # Set figure width to 10 inches
fig_height = fig_width * (
    height / width
)  # Set figure height based on video aspect ratio

# Prepare the plot
fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
img_display = ax.imshow(
    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
)  # Initial display with the first frame
ax.axis("off")  # Hide axes


def update(i):
    ret, img = cap.read()
    if not ret:
        print("No more frames to read or error reading a frame.")
        cap.release()
        plt.close(fig)  # Close the figure to end the animation
        return (img_display,)

    faces = detector(img)
    print(f"Detected {len(faces)} faces")  # Debugging statement

    # Loop through each face detected and apply the predictor
    for face in faces:
        dlib_shape = predictor(img, face)
        shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

        # Debugging output
        print(f"Landmarks for a face: {shape_2d}")

        # Draw rectangles and circles for each face on the image
        cv2.rectangle(
            img,
            (face.left(), face.top()),
            (face.right(), face.bottom()),
            (255, 0, 0),  # Red color for visibility
            2,
            cv2.LINE_AA,
        )
        for s in shape_2d:
            cv2.circle(
                img,
                tuple(s),
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,  # Green color for visibility
            )

    img_display.set_data(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    )  # Update the image displayed
    return (img_display,)


ani = FuncAnimation(fig, update, interval=1, cache_frame_data=False)
plt.show()
