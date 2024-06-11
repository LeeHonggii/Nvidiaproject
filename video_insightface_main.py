import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
import json
import random
from moviepy.editor import VideoFileClip, concatenate_videoclips
from math import radians, degrees, sin, cos, acos
from sympy import symbols, Eq, solve


def process_video(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"No file found at {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Failed to open video file.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center_x, center_y = width // 2, height // 2
    fps = cap.get(cv2.CAP_PROP_FPS)
    is_4k = width >= 1080 and height >= 1080
    display_scale_factor = 0.5 if is_4k else 1

    app = FaceAnalysis(
        allowed_modules=["detection", "landmark_2d_106"],
        providers=["CUDAExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    window_name = "Video"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if display_scale_factor != 1:
        cv2.resizeWindow(
            window_name,
            int(width * display_scale_factor),
            int(height * display_scale_factor),
        )

    frame_count = 0
    face_positions = []
    face_recognitions = []
    eye_endpoint = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_frame_eye_data = []

        if frame_count % 5 == 0:
            faces = app.get(frame)
            current_frame_positions = []
            max_area = 0

            for face in faces:
                bbox = face["bbox"].astype(int)
                x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                area = w * h
                distance_to_center = np.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                current_frame_positions.append((x, y, w, h))
                if "landmark_2d_106" in face:
                    lmk = face["landmark_2d_106"]
                    lmk = np.round(lmk).astype(np.int64)
                    for point in lmk:
                        cv2.circle(
                            frame, tuple(point), 3, (200, 160, 75), 1, cv2.LINE_AA
                        )

                    if distance_to_center < width * 0.25 and area > max_area:
                        max_area = area
                        current_frame_eye_data.append((lmk[35], lmk[93]))

            face_positions.append(current_frame_positions)
            eye_endpoint.append(current_frame_eye_data)

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

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    return face_positions, eye_endpoint, face_recognitions, fps


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


def calculate_eye_distance(eye1, eye2):
    """Calculate the Euclidean distance between two eye points."""
    return np.linalg.norm(np.array(eye1) - np.array(eye2))


def calculate_eye_angle(eye1, eye2):
    """Calculate the angle of the line connecting the eyes with respect to the horizontal."""
    eye1 = np.array(eye1)
    eye2 = np.array(eye2)
    dy = eye2[1] - eye1[1]
    dx = eye2[0] - eye1[0]
    angle = np.degrees(np.arctan2(dy, dx))
    return angle


def feature_vector_from_eyes(eye1, eye2):
    """Generate a feature vector from eye positions, including distance and angle."""
    distance = calculate_eye_distance(eye1, eye2)
    angle = calculate_eye_angle(eye1, eye2)
    return np.array([distance, angle])


def calculate_cosine_similarity(v1, v2):
    """Calculate the cosine similarity between two vectors."""
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


def find_matching_faces(
    face_positions1, eye_endpoints1, face_positions2, eye_endpoints2
):
    matched_faces = []
    min_length = min(len(face_positions1), len(face_positions2))

    for frame_index in range(min_length):
        frame_faces1 = face_positions1[frame_index]
        frame_faces2 = face_positions2[frame_index]

        for i in range(len(frame_faces1)):
            if i >= len(eye_endpoints1[frame_index]):  # Check if the index is valid
                continue

            for j in range(len(frame_faces2)):
                if j >= len(eye_endpoints2[frame_index]):  # Check if the index is valid
                    continue

                x1, y1, w1, h1 = frame_faces1[i]
                x2, y2, w2, h2 = frame_faces2[j]
                iou_score = intersection_over_union(x1, y1, w1, h1, x2, y2, w2, h2)

                if iou_score > 0.5:
                    vector_1 = feature_vector_from_eyes(*eye_endpoints1[frame_index][i])
                    vector_2 = feature_vector_from_eyes(*eye_endpoints2[frame_index][j])
                    cosine_sim = calculate_cosine_similarity(vector_1, vector_2)

                    if cosine_sim > 0.5:
                        matched_faces.append(((frame_index + 1) * 5))
                        print(
                            f"Matched Face at Frame {(frame_index + 1) * 5}: IOU {iou_score:.2f}, Cosine {cosine_sim:.2f}"
                        )

    return matched_faces


def recognition():
    pass


def create_json_structure(
    matched_faces, eye_endpoints1, eye_endpoints2, video_paths, folder_path, fps1, fps2
):
    # Define meta information
    meta_info = {
        "num_stream": len(video_paths),
        "frame_rate": (fps1, fps2),
        "num_frames": 4795.2,  # Example value
        "init_time": 0,
        "duration": 160,  # Example value
        "num_vector_pair": 3,  # Example, adjust based on your actual data
        "num_cross": len(matched_faces),  # Example, adjust based on your actual data
        "first_stream": 0,  # Example value
        "folder_path": folder_path,
    }

    # Define streams
    streams = [{"file": vp, "start": 0, "end": 0} for vp in video_paths]

    time_conversion = 1 / 29.97

    # Define cross_points
    cross_points = []
    for idx, frame_id in enumerate(matched_faces):
        stream_index = idx % len(video_paths)
        if (
            len(eye_endpoints1) > stream_index
            and len(eye_endpoints1[stream_index]) > 0
            and len(eye_endpoints2) > stream_index
            and len(eye_endpoints2[stream_index]) > 0
        ):
            # Ensure that there is at least one eye endpoint data tuple for the current index
            vector_1 = (
                eye_endpoints1[stream_index][0][0].tolist()
                + eye_endpoints1[stream_index][0][1].tolist()
                if isinstance(eye_endpoints1[stream_index][0][0], np.ndarray)
                else list(eye_endpoints1[stream_index][0][0])
                + list(eye_endpoints1[stream_index][0][1])
            )
            vector_2 = (
                eye_endpoints2[stream_index][0][0].tolist()
                + eye_endpoints2[stream_index][0][1].tolist()
                if isinstance(eye_endpoints2[stream_index][0][0], np.ndarray)
                else list(eye_endpoints2[stream_index][0][0])
                + list(eye_endpoints2[stream_index][0][1])
            )
            cross_point = {
                "frame_id": frame_id * time_conversion,
                "next_stream": random.randrange(len(video_paths)),
                "vector_pairs": [{"vector1": vector_1, "vector2": vector_2}],
            }
            cross_points.append(cross_point)

    # Define scene_list for each stream
    scene_list = [
        [100, 500, 1000, 1500],
        [200, 500, 1500, 3000],
        [100, 510, 1000, 1500],
        [400, 500, 1000, 1500],
        [150, 500, 1000, 1500],
        [800, 500, 1000, 1500],
    ]

    # Complete parameter dictionary
    parameter = {
        "meta_info": meta_info,
        "streams": streams,
        "cross_points": cross_points,
        "scene_list": scene_list,
    }

    return parameter


def write_json_file(parameter, output_file):
    with open(output_file, "w") as f:
        json.dump(parameter, f, indent=4)


if __name__ == "__main__":
    folder_path = "./video/"  # Folder path for storing videos
    video_1 = "pose_sync_ive_baddie_1.mp4"
    video_2 = "pose_sync_ive_baddie_2.mp4"
    video_path1 = folder_path + video_1
    video_path2 = folder_path + video_2
    face_positions1, eye_endpoint1, face_recognitions1, fps1 = process_video(
        video_path1
    )
    face_positions2, eye_endpoint2, face_recognitions2, fps2 = process_video(
        video_path2
    )

    matched_faces = find_matching_faces(
        face_positions1, eye_endpoint1, face_positions2, eye_endpoint2
    )
    print(matched_faces)

    video_paths = [
        video_1,
        video_2,
    ]

    # Create the JSON structure
    json_structure = create_json_structure(
        matched_faces,
        eye_endpoint1,
        eye_endpoint2,
        video_paths,
        folder_path,
        fps1,
        fps2,
    )

    # Write the JSON structure to a file
    output_file = "output.json"
    write_json_file(json_structure, output_file)

    print(f"JSON file '{output_file}' has been written with the video analysis data.")
