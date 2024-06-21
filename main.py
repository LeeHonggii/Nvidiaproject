import os
import glob
from concurrent.futures import ProcessPoolExecutor
from pose.pose import process_videos as process_yolo_videos
from pose.pose_similarity import calculate_similarities
from pose.transformation import find_max_transformation_order
from make_json import generate_json, create_combined_video
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


def main():
    with ProcessPoolExecutor() as executor:
        print("영상 분석 시작합니다")
        print("영상 분석 중 입니다")
        video_files = get_video_files(VIDEO_DIR, VIDEO_EXTENSIONS)
        if not video_files:
            print("No video files found in the directory:", VIDEO_DIR)
            return

        csv_video_mapping = process_yolo_videos(video_files)
        if not csv_video_mapping:
            print("YOLO processing did not return any results.")
            return
    print("분석을 종료합니다")
    #print("CSV and Video file mapping (YOLO):")
    #for video, csv in csv_video_mapping.items():
        #print(f"{video} -> {csv}")

    # 자동 매핑된 CSV 파일 목록을 csv_files 리스트에 추가
    csv_files = [csv_video_mapping[video] for video in video_files if video in csv_video_mapping]
    if not csv_files:
        print("No CSV files mapped from the videos.")
        return

    video_file_mapping = {csv: video for video, csv in csv_video_mapping.items()}

    results, verified_matches, frame_similarities, frame_count, best_vectors = calculate_similarities(
        csv_files, WIDTH, HEIGHT, THRESHOLD, POSITION_THRESHOLD, SIZE_THRESHOLD, AVG_SIMILARITY_THRESHOLD
    )

    max_transformation_order = find_max_transformation_order(frame_similarities, frame_count, RANDOM_POINT)

    #print("최대 변환 순서:", max_transformation_order)
    #print("검증된 매칭 결과:", verified_matches)

    json_data = generate_json(max_transformation_order, verified_matches, video_files, csv_files, video_file_mapping,
                              best_vectors)

    with open('output_pose.json', 'w') as f:
        json.dump(json_data, f, indent=4)

    #print("JSON 파일이 생성되었습니다.")
    print("영상 제작 시작합니다")
    # JSON 파일을 기반으로 비디오 합치기
    create_combined_video('output_pose.json', 'combined_video.mp4')
    print("최종 비디오가 생성되었습니다.")


if __name__ == "__main__":
    check_cuda()
    main()
