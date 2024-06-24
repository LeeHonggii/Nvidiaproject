import json
from moviepy.editor import VideoFileClip, concatenate_videoclips

def generate_json(max_transformation_order, verified_matches, video_files, csv_files, video_file_mapping, best_vectors, scene):
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
                        scene
                    ]
    }

    return json_data

def create_combined_video(json_file, output_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    streams = data['streams']
    cross_points = data['cross_points']

    video_clips = {stream['file']: VideoFileClip(stream['file']) for stream in streams}

    combined_clips = []
    current_clip_info = cross_points[0]
    current_clip = video_clips[streams[0]['file']].subclip(0, current_clip_info['time_stamp'])

    combined_clips.append(current_clip)

    for i in range(len(cross_points)):
        cross_point = cross_points[i]
        next_stream_file = streams[cross_point['next_stream']]['file']
        next_clip_start = cross_point['time_stamp']

        if i < len(cross_points) - 1:
            next_clip_end = cross_points[i + 1]['time_stamp']
        else:
            next_clip_end = video_clips[next_stream_file].duration

        next_clip = video_clips[next_stream_file].subclip(next_clip_start, next_clip_end)
        combined_clips.append(next_clip)

    final_clip = concatenate_videoclips(combined_clips, method="compose")
    final_clip.write_videofile(output_file, codec='libx264', fps=24)

