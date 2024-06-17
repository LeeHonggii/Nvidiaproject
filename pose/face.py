import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
import csv
from itertools import combinations
from multiprocessing import Pool, cpu_count

def process_video(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"No file found at {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Failed to open video file.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps else 0

    app = FaceAnalysis(
        allowed_modules=["detection", "landmark_2d_106"],
        providers=["CUDAExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    frame_count = 0
    face_positions = []
    eye_endpoint = []

    LARGE_FACE_AREA = 40000  # 큰 얼굴 면적
    MIN_FACE_AREA = 10000  # 최소 얼굴 면적

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        best_face = None
        min_distance_x = float("inf")
        min_distance_y = float("inf")

        if frame_count % 5 == 0:  # Process every 5th frame
            faces = app.get(frame)

            for face in faces:
                bbox = face["bbox"].astype(int)
                x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                face_area = w * h

                if face_area >= LARGE_FACE_AREA:
                    best_face = face
                    break

            if best_face is None:
                for face in faces:
                    bbox = face["bbox"].astype(int)
                    x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                    face_area = w * h

                    if face_area < MIN_FACE_AREA:
                        continue

                    face_center_x = x + w / 2
                    distance_x = abs(face_center_x - width / 2)

                    if distance_x < min_distance_x:
                        min_distance_x = distance_x
                        best_face_x = face

                for face in faces:
                    bbox = face["bbox"].astype(int)
                    x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                    face_area = w * h

                    if face_area < MIN_FACE_AREA:
                        continue

                    face_center_x = x + w / 2
                    face_center_y = y + h / 2
                    distance_x = abs(face_center_x - width / 2)

                    if distance_x == min_distance_x:
                        distance_y = abs(face_center_y - height / 2)

                        if distance_y < min_distance_y:
                            min_distance_y = distance_y
                            best_face = face

            if best_face:
                bbox = best_face["bbox"].astype(int)
                x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                current_frame_positions = [(x, y, w, h)]

                if "landmark_2d_106" in best_face:
                    lmk = best_face["landmark_2d_106"]
                    lmk = np.round(lmk).astype(np.int64)
                    current_frame_face_data = [lmk.tolist()]
                    eye_point1 = tuple(lmk[35])
                    eye_point2 = tuple(lmk[93])
                    current_frame_eye_data = [(eye_point1, eye_point2)]

                face_positions.append(current_frame_positions)
                eye_endpoint.append(current_frame_eye_data)

    cap.release()
    cv2.destroyAllWindows()

    # Save data to CSV
    input_file_name = os.path.splitext(os.path.basename(video_path))[0]
    output_csv = f"output_{input_file_name}.csv"
    with open(output_csv, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["frame", "x", "y", "w", "h", "eye_point1", "eye_point2"])
        for i, (positions, eye_points) in enumerate(zip(face_positions, eye_endpoint)):
            for (x, y, w, h), (eye_point1, eye_point2) in zip(positions, eye_points):
                csvwriter.writerow([i * 5, x, y, w, h, eye_point1, eye_point2])

    return face_positions, eye_endpoint, fps, total_frames, duration, output_csv

def find_matching_faces(csv_files, min_frame_interval=10):
    all_face_positions = []
    all_eye_endpoints = []
    for csv_file in csv_files:
        face_positions, eye_endpoints = load_csv_data(csv_file)
        all_face_positions.append(face_positions)
        all_eye_endpoints.append(eye_endpoints)

    matched_faces = []
    min_length = min(len(positions) for positions in all_face_positions)
    last_matched_frame = -min_frame_interval

    for frame_index in range(min_length):
        frame_faces = [face_positions[frame_index] for face_positions in all_face_positions]
        frame_eyes = [eye_endpoints[frame_index] for eye_endpoints in all_eye_endpoints]

        if not all(frame_eyes):
            continue

        current_frame = frame_index * 5

        if current_frame - last_matched_frame < min_frame_interval:
            continue

        face_pairs = []
        for i in range(len(frame_faces)):
            for face1 in frame_faces[i]:
                face_pairs.append((i, face1))

        face_pairs_count = len(face_pairs)
        for i in range(face_pairs_count):
            for j in range(i + 1, face_pairs_count):
                index1, face1 = face_pairs[i]
                index2, face2 = face_pairs[j]

                if frame_eyes[index1] and frame_eyes[index2]:
                    x1, y1, w1, h1 = face1
                    x2, y2, w2, h2 = face2
                    iou_score = intersection_over_union(x1, y1, w1, h1, x2, y2, w2, h2)
                    if iou_score > 0.9:
                        matched_faces.append((current_frame, index1, index2))
                        last_matched_frame = current_frame
                        print(f"Matched Face at Frame {current_frame}: IOU {iou_score:.2f}")
                        break
            else:
                continue
            break

    return matched_faces

def load_csv_data(file_path):
    face_positions = []
    eye_endpoints = []
    with open(file_path, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header
        for row in csvreader:
            frame, x, y, w, h, eye_point1, eye_point2 = row
            face_positions.append([(int(x), int(y), int(w), int(h))])
            eye_endpoints.append([
                (
                    (
                        int(eye_point1[1:-1].split(", ")[0]),
                        int(eye_point1[1:-1].split(", ")[1]),
                    ),
                    (
                        int(eye_point2[1:-1].split(", ")[0]),
                        int(eye_point2[1:-1].split(", ")[1]),
                    ),
                )
            ])
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

def process_video_multiprocessing(video_paths):
    with Pool(cpu_count()) as pool:
        results = pool.map(process_video, video_paths)
    return results

