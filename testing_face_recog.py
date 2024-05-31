import cv2
import face_recognition
import numpy as np
import os


def process_video(video_path, output_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"No file found at {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Failed to open video file.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    is_4k = width >= 3840 and height >= 2160
    display_scale_factor = 0.25 if is_4k else 0.5

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    window_name = "Video"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if display_scale_factor != 1:
        cv2.resizeWindow(
            window_name,
            int(width * display_scale_factor),
            int(height * display_scale_factor),
        )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        print("Processing frame...")  # Debug: check if it reaches here

        face_locations = face_recognition.face_locations(frame)
        face_landmarks = face_recognition.face_landmarks(frame)

        if face_locations:
            print(f"Detected {len(face_locations)} faces")

        for face_location, landmarks in zip(face_locations, face_landmarks):
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # for facial_feature in landmarks:
            #     points = landmarks[facial_feature]
            #     for point in points:
            #         cv2.circle(frame, tuple(point), 2, (0, 0, 255), -1)

        out.write(frame)
        display_frame = (
            cv2.resize(
                frame,
                (int(width * display_scale_factor), int(height * display_scale_factor)),
            )
            if display_scale_factor != 1
            else frame
        )
        cv2.imshow(window_name, display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quit signal received")
            break

    # Release resources only after the loop is done
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "video/aespa_forever.mp4"
    output_path = "video/test_aespa_forever.mp4"
    process_video(video_path, output_path)
    print("Video processing complete. Output saved to", output_path)
