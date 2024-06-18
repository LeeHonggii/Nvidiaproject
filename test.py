import torch
import pandas as pd
from pose.similarity import calculate_similarities


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
RANDOM_POINT = 5

# CSV 파일 및 비디오 파일 설정
csv_files = [
    "ive_baddie_1.csv",
    "ive_baddie_2.csv",
    "ive_baddie_3.csv",
    "ive_baddie_4.csv",
    "ive_baddie_5.csv"
]

video_files = [
    "ive_baddie_1.mp4",
    "ive_baddie_2.mp4",
    "ive_baddie_3.mp4",
    "ive_baddie_4.mp4",
    "ive_baddie_5.mp4"
]

# 비디오 파일과 CSV 파일 매핑
video_file_mapping = {
    "ive_baddie_1.csv": "ive_baddie_1.mp4",
    "ive_baddie_2.csv": "ive_baddie_2.mp4",
    "ive_baddie_3.csv": "ive_baddie_3.mp4",
    "ive_baddie_4.csv": "ive_baddie_4.mp4",
    "ive_baddie_5.csv": "ive_baddie_5.mp4"
}

if __name__ == "__main__":
    check_cuda()

    results, max_transformation_order, verified_matches = calculate_similarities(
        csv_files, video_files, video_file_mapping, WIDTH, HEIGHT, THRESHOLD, POSITION_THRESHOLD, SIZE_THRESHOLD,
        AVG_SIMILARITY_THRESHOLD, RANDOM_POINT
    )

    print("최대 변환 순서:", max_transformation_order)
    print("검증된 매칭 결과:", verified_matches)
