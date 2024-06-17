import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
import csv
from multiprocessing import Pool, cpu_count

def initialize_face_analysis():
    app = FaceAnalysis(allowed_modules=["detection", "landmark_2d_106"], providers=["CUDAExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

def process_video_frames(video_path):
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

    app = initialize_face_analysis()

    frame_count = 0
    face_positions = []
    eye_endpoint = []

    LARGE_FACE_AREA = 40000
    MIN_FACE_AREA = 10000

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        best_face = None
        min_distance_x = float("inf")
        min_distance_y = float("inf")

        if frame_count % 5 == 0:
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

                best_face = best_face_x

            if best_face is not None:
                bbox = best_face["bbox"].astype(int)
                x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                keypoints = best_face["landmark_2d_106"].astype(int)
                eye1, eye2 = keypoints[38], keypoints[88]
                face_positions.append((x, y, w, h))
                eye_endpoint.append((eye1, eye2))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return video_path, duration, face_positions, eye_endpoint

def save_results_to_csv(results):
    for result in results:
        video_path, duration, face_positions, eye_endpoint = result
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        csv_filename = f"{base_name}_faces.csv"

        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["video_path", "duration", "face_x", "face_y", "face_w", "face_h", "eye1_x", "eye1_y", "eye2_x", "eye2_y"])

            for i in range(len(face_positions)):
                x, y, w, h = face_positions[i]
                eye1, eye2 = eye_endpoint[i]
                writer.writerow([video_path, duration, x, y, w, h, eye1[0], eye1[1], eye2[0], eye2[1]])

def process_video_multiprocessing(video_files):
    with Pool(cpu_count()) as pool:
        results = pool.map(process_video_frames, video_files)
    save_results_to_csv(results)
    return results
