import cv2
import numpy as np
import os
import insightface
from insightface.app import FaceAnalysis
import time
from scipy.spatial import distance

def extract_landmarks(face):
    """
    Extract the facial landmarks from the face object and flatten them.
    """
    landmarks = face.landmark_2d_106
    return landmarks.flatten() if landmarks is not None else None

def process_video(video_path, output_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"No file found at {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Failed to open video file.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    is_4k = width >= 1080 and height >= 1080
    display_scale_factor = 0.5 if is_4k else 1

    app = FaceAnalysis(allowed_modules=["detection", "landmark_2d_106"], providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    window_name = "Video"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if display_scale_factor != 1:
        cv2.resizeWindow(window_name, int(width * display_scale_factor), int(height * display_scale_factor))

    frame_count = 0
    known_landmarks = []
    known_names = []
    unique_id_counter = 1
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 5 == 0:
            faces = app.get(frame)
            print(f"Frame {frame_count}: Detected {len(faces)} faces")
            found_people = []

            for face in faces:
                bbox = face.bbox.astype(int)
                top, right, bottom, left = max(0, bbox[1]), min(width, bbox[2]), min(height, bbox[3]), max(0, bbox[0])
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                landmarks = extract_landmarks(face)
                if landmarks is not None:
                    if known_landmarks:
                        distances = [distance.euclidean(landmarks, known_landmark) for known_landmark in known_landmarks]
                        best_match_index = np.argmin(distances)
                        min_distance = distances[best_match_index]
                        if min_distance < 20:  # Adjusted threshold for better accuracy
                            name = known_names[best_match_index]
                            print(f"Match found: {name} with distance: {min_distance:.2f}")
                        else:
                            name = f"Unknown_{unique_id_counter}"
                            known_landmarks.append(landmarks)
                            known_names.append(name)
                            unique_id_counter += 1
                            print(f"New person identified: {name} with distance: {min_distance:.2f}")
                    else:
                        name = f"Unknown_{unique_id_counter}"
                        known_landmarks.append(landmarks)
                        known_names.append(name)
                        unique_id_counter += 1
                        print(f"New person identified: {name}")

                    found_people.append(name)
                    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                    # Draw the landmarks
                    lmk = face.landmark_2d_106
                    lmk = np.round(lmk).astype(np.int64)
                    for point in lmk:
                        cv2.circle(frame, tuple(point), 3, (200, 160, 75), 1, cv2.LINE_AA)
                    print(f"Landmarks for the face: {lmk}")

            print(f"Frame {frame_count}: Found people: {found_people}")

        out.write(frame)
        if display_scale_factor != 1:
            display_frame = cv2.resize(frame, (int(width * display_scale_factor), int(height * display_scale_factor)))
            cv2.imshow(window_name, display_frame)
        else:
            cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    total_time = time.time() - start_time
    print(f"Video processing complete. Output saved to {output_path}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print("Known faces:", known_names)

if __name__ == "__main__":
    video_path = "./video/ive_baddie_1.mp4"
    output_path = "./video/output_ive_baddie_1.mp4"
    process_video(video_path, output_path)
