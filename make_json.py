import os
import json
import numpy as np
from .data_loader import load_csv_data  # Assuming you have this function in another file

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
