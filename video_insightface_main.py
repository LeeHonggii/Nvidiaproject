import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
import json


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

    window_name = "Video"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)


    frame_count = 0
    face_positions = []
    face_recognitions = []
    eye_endpoint = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        best_face = None
        min_distance = float("inf")
        max_area = 0

        if frame_count % 5 == 0:  # Process every 5th frame
            faces = app.get(frame)

            for face in faces:
                bbox = face["bbox"].astype(int)
                x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                area = w * h
                distance = (
                    (x + w / 2 - width / 2) ** 2 + (y + h / 2 - height / 2) ** 2
                ) ** 0.5
                if area > max_area or distance < min_distance:
                    max_area = area
                    min_distance = distance
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

                    for point in lmk:
                        cv2.circle(
                            frame, tuple(point), 3, (200, 160, 75), 1, cv2.LINE_AA
                        )
                    cv2.line(frame, eye_point1, eye_point2, (255, 0, 0), 2)

                face_positions.append(current_frame_positions)
                eye_endpoint.append(current_frame_eye_data)
                face_recognitions.append(current_frame_face_data)



            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(face_positions)
    print(total_frames // 5)
    print(len(face_positions))

    return face_positions, eye_endpoint, face_recognitions, fps, total_frames, duration


def position_decision(x, y, w, h, width, height):
    face_center_x = x + w / 2
    face_center_y = y + h / 2

    screen_center_x = width / 2
    screen_center_y = height / 2

    distance = (
        (face_center_x - screen_center_x) ** 2 + (face_center_y - screen_center_y) ** 2
    ) ** 0.5

    area = w * h
    return distance, area


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

        # Ensure there are eye endpoints for both videos at this frame index
        if not eye_endpoints1[frame_index] or not eye_endpoints2[frame_index]:
            print(f"Skipping frame {frame_index*5 + 1} due to missing eye data.")
            continue  # Skip frames without valid eye endpoint data

        for i, face1 in enumerate(frame_faces1):
            for j, face2 in enumerate(frame_faces2):
                if i < len(eye_endpoints1[frame_index]) and j < len(
                    eye_endpoints2[frame_index]
                ):
                    x1, y1, w1, h1 = face1
                    x2, y2, w2, h2 = face2
                    iou_score = intersection_over_union(x1, y1, w1, h1, x2, y2, w2, h2)
                    if iou_score > 0.6:

                        current_frame = (frame_index) * 5
                        matched_faces.append(current_frame)
                        print(
                            f"Matched Face at Frame {current_frame}: IOU {iou_score:.2f}"
                        )

    return matched_faces


def recognition():
    pass


def create_json_structure(
    matched_faces,
    eye_endpoints1,
    eye_endpoints2,
    video_paths,
    folder_path,
    fps1,
    total_frames1,
    duration1,
):
    # Define meta information
    meta_info = {
        "num_stream": len(video_paths),
        "metric": "time",
        "frame_rate": fps1,
        "num_frames": total_frames1,  # Example value
        "init_time": 0,
        "duration": duration1,  # Example value
        "num_vector_pair": 3,  # Example, adjust based on your actual data
        "num_cross": len(matched_faces),  # Example, adjust based on your actual data
        "first_stream": 1,
        "folder_path": folder_path,
    }

    # Define streams
    streams = [{"file": vp, "start": 0, "end": 0} for vp in video_paths]

    time_conversion = 1 / fps1

    next_stream = 0

    # Define cross_points
    cross_points = []

    for idx, frame_id in enumerate(matched_faces):
        index1 = frame_id // 5
        index2 = index1  # Assuming the frames are synchronized in both videos

        if 0 <= index1 < len(eye_endpoints1) and 0 <= index2 < len(eye_endpoints2):
            if eye_endpoints1[index1] and eye_endpoints2[index2]:
                vector_1 = (
                    np.array(eye_endpoints1[index1][0][0]).tolist()
                    + np.array(eye_endpoints1[index1][0][1]).tolist()
                )
                vector_2 = (
                    np.array(eye_endpoints2[index2][0][0]).tolist()
                    + np.array(eye_endpoints2[index2][0][1]).tolist()
                )

                cross_point = {
                    "time_stamp": frame_id * time_conversion,
                    "next_stream": next_stream,
                    "vector_pairs": [{"vector1": vector_1, "vector2": vector_2}],
                }
                cross_points.append(cross_point)
                next_stream = (next_stream + 1) % len(video_paths)
            else:
                print(
                    f"No valid eye endpoint data for frame_id {frame_id} at indices index1={index1}, index2={index2}"
                )
        else:
            print(
                f"Index out of range for frame_id {frame_id}: index1={index1}, index2={index2}, Lengths: {len(eye_endpoints1)}, {len(eye_endpoints2)}"
            )

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
    folder_path = "./data/"  # Folder path for storing videos
    video_1 = "pose_sync_ive_baddie_1.mp4"
    video_2 = "pose_sync_ive_baddie_2.mp4"
    video_path1 = folder_path + video_1
    video_path2 = folder_path + video_2
    (
        face_positions1,
        eye_endpoint1,
        face_recognitions1,
        fps1,
        total_frames1,
        duration1,
    ) = process_video(video_path1)
    (
        face_positions2,
        eye_endpoint2,
        face_recognitions2,
        fps2,
        total_frames2,
        duration2,
    ) = process_video(video_path2)

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
        total_frames1,
        duration1,
    )

    # Write the JSON structure to a file
    output_file = "output.json"
    write_json_file(json_structure, output_file)
    print(eye_endpoint1[102])
    print(eye_endpoint2[102])

    print(f"JSON file '{output_file}' has been written with the video analysis data.")
