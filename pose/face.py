import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from concurrent.futures import ThreadPoolExecutor


def init_face_analysis():
    app = FaceAnalysis(
        allowed_modules=["detection", "landmark_2d_106"],
        providers=["CUDAExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def process_frame(app, video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Failed to open video file.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return frame_number, None, None

    faces = app.get(frame)
    best_face = None
    min_x_distance = float("inf")

    width = frame.shape[1]
    height = frame.shape[0]

    for face in faces:
        bbox = face["bbox"].astype(int)
        x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
        face_center_x = x + w / 2
        x_distance_to_center = abs(face_center_x - width / 2)

        if x_distance_to_center < min_x_distance:
            min_x_distance = x_distance_to_center
            best_face = face

    if best_face:
        bbox = best_face["bbox"].astype(int)
        x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
        current_frame_positions = [(x, y, w, h)]

        if "landmark_2d_106" in best_face:
            lmk = best_face["landmark_2d_106"]
            lmk = np.round(lmk).astype(np.int64)
            eye_point1 = tuple(lmk[35])
            eye_point2 = tuple(lmk[93])
            current_frame_eye_data = [(eye_point1, eye_point2)]

        return frame_number, current_frame_positions, current_frame_eye_data

    return frame_number, None, None


def process_specific_frames(video_path, frame_numbers, app):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_frame, app, video_path, frame) for frame in frame_numbers]
        results = [future.result() for future in futures]

    face_positions = [result[1] for result in results if result[1] is not None]
    eye_endpoints = [result[2] for result in results if result[2] is not None]

    return face_positions, eye_endpoints


def intersection_over_union(x1, y1, w1, h1, x2, y2, w2, h2):
    x1_max = x1 + w1
    y1_max = y1 + h1
    x2_max = x2 + w2
    y2_max = y2 + h2

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h
    rect1_area = w1 * h1
    rect2_area = w2 * h2
    union_area = rect1_area + rect2_area - inter_area

    iou = inter_area / union_area

    return iou


def find_matching_faces(face_positions1, eye_endpoints1, face_positions2, eye_endpoints2):
    matched_faces = []
    min_length = min(len(face_positions1), len(face_positions2))

    for frame_index in range(min_length):
        frame_faces1 = face_positions1[frame_index]
        frame_faces2 = face_positions2[frame_index]

        if not eye_endpoints1[frame_index] or not eye_endpoints2[frame_index]:
            print(f"Skipping frame {frame_index + 1} due to missing eye data.")
            continue

        for i, face1 in enumerate(frame_faces1):
            for j, face2 in enumerate(frame_faces2):
                if i < len(eye_endpoints1[frame_index]) and j < len(eye_endpoints2[frame_index]):
                    x1, y1, w1, h1 = face1
                    x2, y2, w2, h2 = face2
                    iou_score = intersection_over_union(x1, y1, w1, h1, x2, y2, w2, h2)
                    if iou_score > 0.6:
                        current_frame = frame_index + 1
                        matched_faces.append(current_frame)
                        print(f"Matched Face at Frame {current_frame}: IOU {iou_score:.2f}")

    return matched_faces
