import cv2
import numpy as np
import os
import insightface
from insightface.app import FaceAnalysis
import time


def process_video(video_path, output_path):

    # Check if the video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"No file found at {video_path}")

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Failed to open video file.")

    # Check video resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    is_4k = width >= 3840 and height >= 2160  # Typical 4K resolution check
    display_scale_factor = 0.25 if is_4k else 1  # Scale display if 4K

    start_time = time.time()

    # Initialize the FaceAnalysis app with landmark detection
    app = FaceAnalysis(allowed_modules=["detection", "landmark_2d_106"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Video writer for output without resizing
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec used to create the output video
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Set up window for resized display if necessary
    window_name = "Video"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if display_scale_factor != 1:
        cv2.resizeWindow(
            window_name,
            int(width * display_scale_factor),
            int(height * display_scale_factor),
        )
    frame_count = 0
    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 5 == 0:
            # Perform face detection and landmark detection
            faces = app.get(frame)
            for face in faces:
                bbox = face["bbox"].astype(int)
                cv2.rectangle(
                    frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2
                )  # Draw bounding box
                lmk = face["landmark_2d_106"]
                if lmk is not None:
                    lmk = np.round(lmk).astype(np.int16)
                    for point in lmk:
                        cv2.circle(
                            frame, tuple(point), 3, (200, 160, 75), 1, cv2.LINE_AA
                        )  # Draw landmarks

            # Save the full resolution frame
            out.write(frame)

            # Display the frame, scaled if it's 4K
            if display_scale_factor != 1:
                display_frame = cv2.resize(
                    frame,
                    (
                        int(width * display_scale_factor),
                        int(height * display_scale_factor),
                    ),
                )
                cv2.imshow(window_name, display_frame)
            else:
                cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
                break

    # Release resources
    cap.release()
    out.release()
    end_time = time.time()
    cv2.destroyAllWindows()
    total_time = end_time - start_time
    print(f"Video processing complete. Output saved to {output_path}")
    print(f"Total processing time: {total_time:.2f} seconds")

    return


if __name__ == "__main__":
    video_path = "C:/Users/hancomtst/Desktop/Nvidiaproject/video/ive_baddie_1.mp4"  # Update this path
    output_path = "./output_ive_baddie_1.mp4"  # Update this path if needed
    process_video(video_path, output_path)
    print("Video processing complete. Output saved to", output_path)
