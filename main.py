import os
import asyncio
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
RANDOM_POINT = 5

video_files = [
    "ive_baddie_1.mp4",
    "ive_baddie_2.mp4",
    "ive_baddie_3.mp4",
    "ive_baddie_4.mp4",
    "ive_baddie_5.mp4"
]

async def main():
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor() as executor:
        yolo_future = loop.run_in_executor(executor, process_yolo_videos, video_files)
        # face_future = loop.run_in_executor(executor, process_video_multiprocessing, video_files)

        # csv_video_mapping, face_processing_results = await asyncio.gather(yolo_future, face_future)
        csv_video_mapping = await yolo_future

    print("CSV and Video file mapping (YOLO):")
    for video, csv in csv_video_mapping.items():
        print(f"{video} -> {csv}")

    # 자동 매핑된 CSV 파일 목록을 csv_files 리스트에 추가
    csv_files = [csv_video_mapping[video] for video in video_files]
    video_file_mapping = {csv: video for video, csv in csv_video_mapping.items()}

    # print("Face processing results:")
    # for result in face_processing_results:
    #     print(result)

    results, verified_matches, frame_similarities, frame_count, best_vectors = calculate_similarities(
        csv_files, WIDTH, HEIGHT, THRESHOLD, POSITION_THRESHOLD, SIZE_THRESHOLD, AVG_SIMILARITY_THRESHOLD
    )

    max_transformation_order = find_max_transformation_order(frame_similarities, frame_count, 5)

    print("최대 변환 순서:", max_transformation_order)
    print("검증된 매칭 결과:", verified_matches)

    json_data = generate_json(max_transformation_order, verified_matches, video_files, csv_files, video_file_mapping,
                              best_vectors)

    with open('output_pose.json', 'w') as f:
        json.dump(json_data, f, indent=4)

    print("JSON 파일이 생성되었습니다.")

    # JSON 파일을 기반으로 비디오 합치기
    create_combined_video('output_pose.json', 'combined_video.mp4')
    print("최종 비디오가 생성되었습니다.")

if __name__ == "__main__":
    check_cuda()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())