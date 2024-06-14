import os
from tqdm import tqdm
from video_processor import process_videos
from pose.similarity import calculate_similarities

# 하이퍼파라미터 설정
WIDTH = 1920
HEIGHT = 1080
# 포즈 유사도
THRESHOLD = 8
# 위치
POSITION_THRESHOLD = 0.05
# 크기
SIZE_THRESHOLD = 0.05
# 평균 유사도
AVG_SIMILARITY_THRESHOLD = 0.5

if __name__ == "__main__":
    # 비디오 파일 목록
    video_files = [
        "ive_baddie_1.mp4",
        "ive_baddie_2.mp4",
        "ive_baddie_3.mp4",
        "ive_baddie_4.mp4",
        "ive_baddie_5.mp4"
    ]

    # 비디오 파일을 처리하고 CSV 파일 매핑을 가져옴
    csv_video_mapping = process_videos(video_files)
    print("CSV and Video file mapping:")
    for video, csv in csv_video_mapping.items():
        print(f"{video} -> {csv}")

    # CSV 파일 목록 및 비디오 파일 매핑
    csv_files = list(csv_video_mapping.values())
    video_file_mapping = {csv: video for video, csv in csv_video_mapping.items()}

    # 유사도 계산 수행
    results, max_transformation_order, verified_matches = calculate_similarities(
        csv_files, video_files, video_file_mapping, WIDTH, HEIGHT, THRESHOLD, POSITION_THRESHOLD, SIZE_THRESHOLD, AVG_SIMILARITY_THRESHOLD
    )
    print("최대 변환 순서:", max_transformation_order)
    print("검증된 매칭 결과:", verified_matches)
