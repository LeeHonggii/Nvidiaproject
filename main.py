import os
import asyncio
from concurrent.futures import ProcessPoolExecutor
from pose.pose import process_videos as process_yolo_videos
from pose.similarity import calculate_similarities
# from pose.face import process_video_multiprocessing, process_video_frames
import torch


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

    # print("Face processing results:")
    # for result in face_processing_results:
    #     print(result)

    csv_files = list(csv_video_mapping.values())
    video_file_mapping = {csv: video for video, csv in csv_video_mapping.items()}

    results, max_transformation_order, verified_matches = calculate_similarities(
        csv_files, video_files, video_file_mapping, WIDTH, HEIGHT, THRESHOLD, POSITION_THRESHOLD, SIZE_THRESHOLD,
        AVG_SIMILARITY_THRESHOLD, RANDOM_POINT
    )
    print("최대 변환 순서:", max_transformation_order)
    print("검증된 매칭 결과:", verified_matches)


if __name__ == "__main__":
    check_cuda()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
