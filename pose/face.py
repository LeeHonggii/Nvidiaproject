import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis


def initialize_face_analysis():
    app = FaceAnalysis(
        allowed_modules=["detection", "landmark_2d_106"],
        providers=["CUDAExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def open_video(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"No file found at {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Failed to open video file.")
    return cap


def get_video_properties(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps else 0
    return width, height, fps, total_frames, duration


def process_frame(frame, app, width, height, prev_bbox):
    best_face = None
    min_distance = float("inf")
    min_x_distance = float("inf")

    faces = app.get(frame)
    for face in faces:
        bbox = face["bbox"].astype(int)
        x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
        face_center_x = x + w / 2
        face_center_y = y + h / 2

        x_distance_to_center = abs(face_center_x - width / 2)

        if prev_bbox is not None:
            prev_x, prev_y, prev_w, prev_h = prev_bbox
            distance = np.sqrt(
                (face_center_x - (prev_x + prev_w / 2)) ** 2
                + (face_center_y - (prev_y + prev_h / 2)) ** 2
            )
        else:
            distance = np.sqrt(
                (face_center_x - width / 2) ** 2
                + (face_center_y - height / 2) ** 2
            )

        if x_distance_to_center < min_x_distance or distance < min_distance:
            min_x_distance = x_distance_to_center
            min_distance = distance
            best_face = face

    return best_face


def draw_landmarks_and_eye_line(frame, best_face):
    lmk = best_face["landmark_2d_106"]
    lmk = np.round(lmk).astype(np.int64)
    eye_point1 = tuple(lmk[35])
    eye_point2 = tuple(lmk[93])

    for point in lmk:
        cv2.circle(frame, tuple(point), 3, (200, 160, 75), 1, cv2.LINE_AA)
    cv2.line(frame, eye_point1, eye_point2, (255, 0, 0), 2)

    return lmk.tolist(), (eye_point1, eye_point2)


def calculate_cosine_similarity(v1, v2):
    v1 = np.asarray(v1).flatten()
    v2 = np.asarray(v2).flatten()

    if np.all(v1 == 0) or np.all(v2 == 0):
        return 0.0  # Avoid division by zero

    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0  # Avoid division by zero if norm is zero

    return dot_product / (norm_v1 * norm_v2)


def process_video_frames(video_path, frame_numbers):
    cap = open_video(video_path)
    width, height, fps, total_frames, duration = get_video_properties(cap)
    app = initialize_face_analysis()

    window_name = "Video"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frames_data = []
    prev_bbox = None

    for frame_number in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            break

        best_face = process_frame(frame, app, width, height, prev_bbox)
        if best_face:
            bbox = best_face["bbox"].astype(int)
            x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            prev_bbox = (x, y, w, h)

            if "landmark_2d_106" in best_face:
                lmk, eye_points = draw_landmarks_and_eye_line(frame, best_face)
                frames_data.append({
                    "frame_number": frame_number,
                    "bbox": (x, y, w, h),
                    "landmarks": lmk,
                    "eye_points": eye_points
                })

            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    return frames_data


def compare_faces(frames_data):
    if len(frames_data) != 2:
        raise ValueError("Exactly two frames must be processed for comparison")

    landmarks1 = frames_data[0]["landmarks"]
    landmarks2 = frames_data[1]["landmarks"]

    similarity = calculate_cosine_similarity(landmarks1, landmarks2)
    threshold = 0.8  # Example threshold for determining if faces match

    return similarity >= threshold, similarity
