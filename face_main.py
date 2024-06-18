import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
import json
import csv
import time
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

    window_name = "Video"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Define the x and y coordinates for the vertical and horizontal lines
    line1_x = int(width * 0.35)
    line2_x = int(width * 0.65)
    line1_y = int(height * 0.25)
    line2_y = int(height * 0.6)
    target_x = (line1_x + line2_x) // 2
    target_y = (line1_y + line2_y) // 2

    frame_count = 0
    face_positions = []
    face_recognitions = []
    eye_endpoint = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        display_frame = frame.copy()

        # Draw vertical and horizontal lines
        cv2.line(display_frame, (line1_x, 0), (line1_x, height), (255, 255, 0), 2)
        cv2.line(display_frame, (line2_x, 0), (line2_x, height), (255, 255, 0), 2)
        cv2.line(display_frame, (0, line1_y), (width, line1_y), (255, 255, 0), 2)
        cv2.line(display_frame, (0, line2_y), (width, line2_y), (255, 255, 0), 2)

        best_face = None
        min_distance = float("inf")

        faces = app.get(frame)

        for face in faces:
            bbox = face["bbox"].astype(int)
            face_center_x = bbox[0] + (bbox[2] - bbox[0]) // 2
            face_center_y = bbox[1] + (bbox[3] - bbox[1]) // 2

            if line1_x <= face_center_x <= line2_x and line1_y <= face_center_y <= line2_y:
                cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                distance = abs(face_center_x - target_x) + abs(face_center_y - target_y)

                if best_face is None or distance < min_distance or (abs(distance - min_distance) < 10 and (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) > (best_face["bbox"][2] - best_face["bbox"][0]) * (best_face["bbox"][3] - best_face["bbox"][1])):
                    min_distance = distance
                    best_face = face
            else:
                cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1)

        if best_face:
            bbox = best_face["bbox"].astype(int)
            cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 3)
            if "landmark_2d_106" in best_face:
                lmk = best_face["landmark_2d_106"]
                lmk = np.round(lmk).astype(np.int64)
                for point in lmk:
                    cv2.circle(display_frame, tuple(point), 2, (0, 0, 255), -1, cv2.LINE_AA)
                current_frame_positions = [(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])]
                current_frame_face_data = [lmk.tolist()]
                eye_point1 = tuple(lmk[35])
                eye_point2 = tuple(lmk[93])
                current_frame_eye_data = [(eye_point1, eye_point2)]

                face_positions.append(current_frame_positions)
                eye_endpoint.append(current_frame_eye_data)
                face_recognitions.append(current_frame_face_data)

        cv2.imshow(window_name, display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    input_file_name = os.path.splitext(os.path.basename(video_path))[0]
    output_csv = f"output_{input_file_name}.csv"
    with open(output_csv, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["frame", "x", "y", "w", "h", "eye_point1", "eye_point2"])
        for i, (positions, eye_points) in enumerate(zip(face_positions, eye_endpoint)):
            for (x, y, w, h), (eye_point1, eye_point2) in zip(positions, eye_points):
                csvwriter.writerow([i * 5, x, y, w, h, eye_point1, eye_point2])

    return (
        face_positions,
        eye_endpoint,
        face_recognitions,
        fps,
        total_frames,
        duration,
        output_csv,
    )

def load_csv_data(file_path):
    face_positions = []
    eye_endpoints = []
    with open(file_path, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header
        for row in csvreader:
            frame, x, y, w, h, eye_point1, eye_point2 = row
            face_positions.append([(int(x), int(y), int(w), int(h))])
            eye_endpoints.append(
                [
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
                ]
            )
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


def find_matching_faces(csv_files):
    all_face_positions = []
    all_eye_endpoints = []
    for csv_file in csv_files:
        face_positions, eye_endpoints = load_csv_data(csv_file)
        all_face_positions.append(face_positions)
        all_eye_endpoints.append(eye_endpoints)

    matched_faces = []
    min_length = min(len(positions) for positions in all_face_positions)

    last_matched_frame = -float("inf")

    for frame_index in range(min_length):
        frame_faces = [
            face_positions[frame_index] for face_positions in all_face_positions
        ]
        frame_eyes = [eye_endpoints[frame_index] for eye_endpoints in all_eye_endpoints]

        if not all(frame_eyes):
            continue  # Skip frames without valid eye endpoint data

        current_frame = frame_index 
        
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

                    area1 = w1 * h1
                    area2 = w2 * h2

                    if area1 > 0 and area2 > 0:
                        area_ratio = max(area1, area2) / min(area1, area2)
                        if area_ratio > 1.5:
                            continue  # Skip this frame if area size difference is more than 1.5

                    iou_score = intersection_over_union(x1, y1, w1, h1, x2, y2, w2, h2)
                    if iou_score > 0.6:
                        matched_faces.append((current_frame, index1, index2))
                        last_matched_frame = current_frame
                        print(
                            f"Matched Face at Frame {current_frame}: IOU {iou_score:.2f}"
                        )
                        break  # Exit loop once a match is found in this frame
            else:
                continue
            break

    return matched_faces


def create_json_structure(
    matched_faces,
    csv_files,
    video_paths,
    folder_path,
    fps,
    total_frames,
    duration,
    scene_list,
):
    all_face_positions = []
    all_eye_endpoints = []
    for csv_file in csv_files:
        face_positions, eye_endpoints = load_csv_data(csv_file)
        all_face_positions.append(face_positions)
        all_eye_endpoints.append(eye_endpoints)

    # Define meta information
    meta_info = {
        "num_stream": len(video_paths),
        "metric": "time",
        "frame_rate": fps[0],
        "num_frames": total_frames[0],
        "init_time": 0,
        "duration": duration[0],
        "num_vector_pair": 3,
        "num_cross": len(matched_faces),
        "first_stream": 1,
        "folder_path": folder_path,
    }

    # Define streams
    streams = [
        {"file": os.path.basename(vp), "start": 0, "end": 0} for vp in video_paths
    ]

    time_conversion = 1 / fps[0]

    next_stream = 0

    # Define cross_points
    cross_points = []

    for idx, (frame_id, i, j) in enumerate(matched_faces):
        index1 = frame_id // 5
        if 0 <= index1 < len(all_eye_endpoints[i]) and 0 <= index1 < len(
            all_eye_endpoints[j]
        ):
            if all_eye_endpoints[i][index1] and all_eye_endpoints[j][index1]:
                vector_1 = (
                    np.array(all_eye_endpoints[i][index1][0][0]).tolist()
                    + np.array(all_eye_endpoints[i][index1][0][1]).tolist()
                )
                vector_2 = (
                    np.array(all_eye_endpoints[j][index1][0][0]).tolist()
                    + np.array(all_eye_endpoints[j][index1][0][1]).tolist()
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
                    f"No valid eye endpoint data for frame_id {frame_id} at index1={index1}"
                )

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


def frame_difference_detection(
    video_path, threshold=20, resize_factor=1, aggregation_window=18
):
    frame_list = []
    cap = cv2.VideoCapture(video_path)
    transition_count = 0
    total_frames = 0
    added_frames = 0  # To keep track of how many times we've added an extra frame

    ret, frame1 = cap.read()
    if not ret:
        print("Error: Failed to read video file.")
        return 0

    # Resize frame to speed up processing
    frame1 = cv2.resize(frame1, (0, 0), fx=resize_factor, fy=resize_factor)
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    last_cut_frame = (
        -aggregation_window
    )  # Initialize to a value outside possible frame index

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break

        frame2 = cv2.resize(frame2, (0, 0), fx=resize_factor, fy=resize_factor)
        total_frames += 1
        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Compute the absolute difference between the current frame and the previous frame
        frame_diff = cv2.absdiff(prev_gray, gray)
        mean_diff = np.mean(frame_diff)

        # Correct frame count at every 1000th frame
        if total_frames % 1000 == 0 and total_frames // 1000 > added_frames:
            total_frames += 1
            added_frames += 1  # Update the counter for added frames

        # If the average intensity difference exceeds the threshold, it's likely a cut
        if mean_diff > threshold:
            # Aggregate close detections as a single cut
            if total_frames - last_cut_frame > aggregation_window:
                transition_count += 1
                last_cut_frame = total_frames

                # print(
                #     f"Cut detected at frame {total_frames} with average frame difference {mean_diff}"
                # )

                frame_list.append(total_frames)
                print((total_frames) * (1 / 29.97))
        prev_gray = gray

    cap.release()
    return frame_list


def process_video_multiprocessing(video_paths):
    with Pool(cpu_count()) as pool:
        results = pool.map(process_video, video_paths)
    return results


if __name__ == "__main__":

    start_time = time.time()

    folder_path = "./data/"  # Folder path for storing videos
    video_files = [
        "ive_baddie_1.mp4",
        "ive_baddie_2.mp4",
        "ive_baddie_3.mp4",
        "ive_baddie_4.mp4",
        "ive_baddie_5.mp4",
        "ive_baddie_6.mp4",
    ]
    video_paths = [os.path.join(folder_path, video_file) for video_file in video_files]
    results = process_video_multiprocessing(video_paths)
    csv_files = []
    # csv_files = [#     "output_ive_baddie_1.csv",#     "output_ive_baddie_2.csv",#     "output_ive_baddie_3.csv",#     "output_ive_baddie_4.csv",#     "output_ive_baddie_5.csv",#     "output_ive_baddie_6.csv",# ]
    fps_list = []
    # fps_list = [29.97002997002997]
    total_frames_list = []
    # total_frames_list = [4817]
    duration_list = []
    # duration_list = [160.72723333333334]

    for result in results:
        (
            face_positions,
            eye_endpoint,
            face_recognitions,
            fps,
            total_frames,
            duration,
            output_csv,
        ) = result
        csv_files.append(output_csv)
        fps_list.append(fps)
        total_frames_list.append(total_frames)
        duration_list.append(duration)

    matched_faces = find_matching_faces(csv_files)
    print(matched_faces)

    finished_time = time.time()

    processed_time = finished_time - start_time
    print("------------------------------------------------------------")
    print("processed_time : ", processed_time)

    scene_list = []

    # for i in range(0, len(video_files)):
    #     frame_detected_list = []
    #     frame_detected_list = frame_difference_detection(video_paths[i])
    #     scene_list.append(frame_detected_list)

    # Create the JSON structure
    json_structure = create_json_structure(
        matched_faces,
        csv_files,
        video_paths,
        folder_path,
        fps_list,
        total_frames_list,
        duration_list,
        scene_list,
    )

    # Write the JSON structure to a file
    output_file = "output.json"
    write_json_file(json_structure, output_file)

    print(f"JSON file '{output_file}' has been written with the video analysis data.")
