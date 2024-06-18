import pandas as pd
import json

def generate_json(max_transformation_order, verified_matches, video_files, csv_files, video_file_mapping, best_vectors):
    num_streams = len(video_files)
    fps = [29.97] * num_streams  # 각 비디오의 fps (임시 값)
    total_frames = [774] * num_streams  # 각 비디오의 총 프레임 수 (임시 값)
    duration = [25.8258] * num_streams  # 각 비디오의 길이 (임시 값)

    meta_info = {
        "num_stream": num_streams,
        "metric": "time",
        "frame_rate": fps[0],
        "num_frames": total_frames[0],
        "init_time": 0,
        "duration": duration[0],
        "num_vector_pair": 3,
        "num_cross": len(max_transformation_order),
        "first_stream": 1,
        "folder_path": "",  # 폴더 경로 필요 없음
    }

    streams = [{"file": video_file, "start": 0, "end": 0} for video_file in video_files]

    cross_points = []
    for frame, start_file, end_file in max_transformation_order:
        time_stamp = frame / fps[0]  # 전환되는 프레임 값을 적음
        next_stream = video_files.index(video_file_mapping[end_file])
        vector1, vector2 = best_vectors.get((frame, start_file, end_file), ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]))
        vector_pairs = [
            {
                "vector1": vector1,
                "vector2": vector2
            }
        ]
        cross_points.append({
            "time_stamp": time_stamp,
            "next_stream": next_stream,
            "vector_pairs": vector_pairs
        })

    json_data = {
        "meta_info": meta_info,
        "streams": streams,
        "cross_points": cross_points,
        "scene_list": [
        [100, 500, 1000, 1500],
        [200, 500, 1500, 3000],
        [100, 510, 1000, 1500],
        [400, 500, 1000, 1500],
        [150, 500, 1000, 1500],
        [800, 500, 1000, 1500],
    ]
    }

    return json_data
