import os
import glob
from concurrent.futures import ProcessPoolExecutor
from pose.pose import process_videos as process_yolo_videos
from pose.pose_similarity import calculate_similarities
from pose.transformation import find_max_transformation_order
from make_json import generate_json, create_combined_video
from pose.face import process_video_multiprocessing, find_matching_faces, process_matches, save_verified_matches, frame_difference_detection
import torch
import json


def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available. Device:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available.")


# 하이퍼파라미터 설정
WIDTH = 1920
HEIGHT = 1080
THRESHOLD = 8
POSITION_THRESHOLD = 0.05
SIZE_THRESHOLD = 0.05
AVG_SIMILARITY_THRESHOLD = 0.5
RANDOM_POINT = 10

# 절대 경로를 사용하도록 수정
VIDEO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))  # 비디오 파일들이 있는 폴더 경로
VIDEO_EXTENSIONS = ['mp4', 'avi', 'mkv', 'mov']  # 지원하는 비디오 파일 확장자


def get_video_files(video_dir, extensions):
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, f"*.{ext}")))
    return video_files


def find_intersections(face_verified_matches, frame_similarities):
    # Convert face_verified_matches into a set for faster lookup
    face_match_set = {
        (frame_number, filename1, filename2)
        for frame_number, filename1, filename2 in face_verified_matches
    }

    # Initialize the updated_frame_similarities dictionary to store matching frame details
    updated_frame_similarities = {}
    for file, similarities in frame_similarities.items():
        for (frame_num, (file1, file2)) in similarities:
            file1_base = os.path.splitext(file1)[0]
            file2_base = os.path.splitext(file2)[0]
            if (frame_num, file1_base, file2_base) in face_match_set or (frame_num, file2_base, file1_base) in face_match_set:
                if frame_num not in updated_frame_similarities:
                    updated_frame_similarities[frame_num] = []
                if (file1, file2) not in updated_frame_similarities[frame_num]:
                    updated_frame_similarities[frame_num].append((file1, file2))
    return updated_frame_similarities


def main():
    check_cuda()
    with ProcessPoolExecutor() as executor:
        print("영상 분석 시작합니다")
        video_files = get_video_files(VIDEO_DIR, VIDEO_EXTENSIONS)
        if not video_files:
            print("No video files found in the directory:", VIDEO_DIR)
            return

        csv_video_mapping = process_yolo_videos(video_files)
        csv_face_mapping = process_video_multiprocessing(video_files)
        if not csv_video_mapping:
            print("YOLO processing did not return any results.")
            return
        if not csv_face_mapping:
            print("Face Detection did not return any results.")

    print("분석을 종료합니다")

    csv_files = [csv_video_mapping[video] for video in video_files if video in csv_video_mapping]
    if not csv_files:
        print("No CSV files mapped from the videos.")
        return

    csv_face_files = []
    for video in video_files:
        base_name = os.path.basename(video).split(".")[0]
        csv_path = f"output_{base_name}.csv"
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            csv_face_files.append(csv_path)
        else:
            print(f"CSV file {csv_path} is missing or empty for video {video}")
    if not csv_face_files:
        print("No valid CSV files available for face matching.")
        return

    matched_faces = find_matching_faces(csv_face_files)

    simple_verified_matches, detailed_verified_matches = process_matches(matched_faces, video_files)
    save_verified_matches(detailed_verified_matches)

    print("Face Verified Matches:", simple_verified_matches)

    video_file_mapping = {csv: video for video, csv in csv_video_mapping.items()}

    results, verified_matches, frame_similarities, frame_count, best_vectors = calculate_similarities(
        csv_files, WIDTH, HEIGHT, THRESHOLD, POSITION_THRESHOLD, SIZE_THRESHOLD, AVG_SIMILARITY_THRESHOLD
    )

    print("Frame Similarities:", frame_similarities)

    n_frame_similarities = find_intersections(simple_verified_matches, frame_similarities)

    print("Updated Frame Similarities:", n_frame_similarities)

    max_transformation_order = find_max_transformation_order(
        n_frame_similarities, frame_count, RANDOM_POINT
    )
    print("Max Transformation Order:", max_transformation_order)

    detected_scene = frame_difference_detection(video_files)

    json_data = generate_json(max_transformation_order, verified_matches, video_files, csv_files, video_file_mapping,
                              best_vectors, detected_scene)

    with open('output_pose.json', 'w') as f:
        json.dump(json_data, f, indent=4)

    print("영상 제작 시작합니다")

    json_file = "output_pose.json"
    #with open(json_file, "r", encoding="utf-8") as file:
    #    json_string_from_file = file.read()
    #run_video_editor(json_string_from_file)
    #print("최종 비디오가 생성되었습니다.")


if __name__ == "__main__":
    main()